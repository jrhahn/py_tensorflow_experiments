import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from base_line import Baseline
from data_types.training_result import TrainingResult
from data_types.training_set import TrainingSet
from window_generator import WindowGenerator

MAX_EPOCHS = 20


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    wv = df['wv (m/s)']
    bad_wv = wv == -9999.0
    wv[bad_wv] = 0.0

    max_wv = df['max. wv (m/s)']
    bad_max_wv = max_wv == -9999.0
    max_wv[bad_max_wv] = 0.0

    # The above inplace edits are reflected in the DataFrame.
    # df['wv (m/s)'].min()

    return df


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    wv = df.pop('wv (m/s)')
    max_wv = df.pop('max. wv (m/s)')

    # Convert to radians.
    wd_rad = df.pop('wd (deg)') * np.pi / 180

    # Calculate the wind x and y components.
    df['Wx'] = wv * np.cos(wd_rad)
    df['Wy'] = wv * np.sin(wd_rad)

    # Calculate the max wind x and y components.
    df['max Wx'] = max_wv * np.cos(wd_rad)
    df['max Wy'] = max_wv * np.sin(wd_rad)

    date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
    timestamp_s = date_time.map(pd.Timestamp.timestamp)

    day = 24 * 60 * 60
    year = (365.2425) * day

    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    return df


def get_data() -> pd.DataFrame:
    zip_path = tf.keras.utils.get_file(
        origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
        fname='jena_climate_2009_2016.csv.zip',
        extract=True)
    csv_path, _ = os.path.splitext(zip_path)

    df = pd.read_csv(csv_path)
    # Slice [start:stop:step], starting from index 5 take every 6th record.
    df = df[5::6]

    # date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')

    # return csv_path

    return df


def prepare_sets(df: pd.DataFrame) -> TrainingSet:
    n = len(df)
    train_df = df[0:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.9)]
    test_df = df[int(n * 0.9):]

    num_features = df.shape[1]

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    df_std = (df - train_mean) / train_std
    df_std = df_std.melt(var_name='Column', value_name='Normalized')
    plt.figure(figsize=(12, 6))
    # ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
    # _ = ax.set_xticklabels(df.keys(), rotation=90)

    return TrainingSet(
        training=train_df,
        test=test_df,
        validation=val_df
    )


# def setup_window_generators(
#         training_set: TrainingSet
# ):
#     w1 = WindowGenerator(
#         input_width=24,
#         label_width=1,
#         shift=24,
#         label_columns=['T (degC)'],
#         train_df=training_set.training,
#         val_df=training_set.evaluation,
#         test_df=training_set.test
#     )
#
#     w2 = WindowGenerator(
#         input_width=6,
#         label_width=1,
#         shift=1,
#         label_columns=['T (degC)'],
#         train_df=training_set.training,
#         val_df=training_set.evaluation,
#         test_df=training_set.test
#     )
#
#     # Stack three slices, the length of the total window.
#     example_window = tf.stack(
#         [
#             np.array(training_set.training[:w2.total_window_size]),
#             np.array(training_set.training[100:100 + w2.total_window_size]),
#             np.array(training_set.training[200:200 + w2.total_window_size])
#         ]
#     )
#
#     example_inputs, example_labels = w2.split_window(example_window)
#
#     print('All shapes are: (batch, time, features)')
#     print(f'Window shape: {example_window.shape}')
#     print(f'Inputs shape: {example_inputs.shape}')
#     print(f'Labels shape: {example_labels.shape}')


def test_baseline(
        training_set: TrainingSet
) -> TrainingResult:
    column_indices = {name: i for i, name in enumerate(training_set.training.columns)}

    baseline = Baseline(label_index=column_indices['T (degC)'])

    baseline.compile(
        loss=tf.losses.MeanSquaredError(),
        metrics=[tf.metrics.MeanAbsoluteError()]
    )

    single_step_window = WindowGenerator(
        input_width=1,
        label_width=1,
        shift=1,
        training_set=training_set,
        label_columns=['T (degC)']
    )

    result = TrainingResult(
        validation_performance=baseline.evaluate(single_step_window.val),
        performance=baseline.evaluate(single_step_window.test, verbose=0)
    )

    wide_window = WindowGenerator(
        input_width=24,
        label_width=24,
        shift=1,
        label_columns=['T (degC)'],
        training_set=training_set
    )

    wide_window.plot(baseline)

    return result


def compile_and_fit(model, window, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history


def test_linear(
        training_set: TrainingSet
) -> TrainingResult:
    ## LINEAR
    linear = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1)
    ])

    single_step_window = WindowGenerator(
        input_width=1,
        label_width=1,
        shift=1,
        training_set=training_set,
        label_columns=['T (degC)']
    )

    print('Input shape:', single_step_window.example[0].shape)
    print('Output shape:', linear(single_step_window.example[0]).shape)

    compile_and_fit(linear, single_step_window)

    wide_window = WindowGenerator(
        input_width=24,
        label_width=24,
        shift=1,
        label_columns=['T (degC)'],
        training_set=training_set
    )
    wide_window.plot(linear)

    return TrainingResult(
        performance=linear.evaluate(single_step_window.test, verbose=0),
        validation_performance=linear.evaluate(single_step_window.val)
    )


def test_dense(
        training_set: TrainingSet
) -> TrainingResult:
    dense = tf.keras.Sequential([
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=1)
    ])

    single_step_window = WindowGenerator(
        input_width=1,
        label_width=1,
        shift=1,
        training_set=training_set,
        label_columns=['T (degC)']
    )

    compile_and_fit(dense, single_step_window)

    return TrainingResult(
        validation_performance=dense.evaluate(single_step_window.val),
        performance=dense.evaluate(single_step_window.test, verbose=0)
    )


def test_multi_step_dense(
        training_set: TrainingSet,
        CONV_WIDTH: int = 3
) -> TrainingResult:
    conv_window = WindowGenerator(
        input_width=CONV_WIDTH,
        label_width=1,
        shift=1,
        label_columns=['T (degC)'],
        training_set=training_set
    )

    multi_step_dense = tf.keras.Sequential([
        # Shape: (time, features) => (time*features)
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1),
        # Add back the time dimension.
        # Shape: (outputs) => (1, outputs)
        tf.keras.layers.Reshape([1, -1]),
    ])

    print('Input shape:', conv_window.example[0].shape)
    print('Output shape:', multi_step_dense(conv_window.example[0]).shape)

    compile_and_fit(multi_step_dense, conv_window)

    return TrainingResult(
        validation_performance=multi_step_dense.evaluate(conv_window.val),
        performance=multi_step_dense.evaluate(conv_window.test, verbose=0)
    )


def test_multi_step_conv_net(
        training_set: TrainingSet,
        CONV_WIDTH: int = 3
) -> TrainingResult:
    conv_model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32,
                               kernel_size=(CONV_WIDTH,),
                               activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1),
    ])

    conv_window = WindowGenerator(
        input_width=CONV_WIDTH,
        label_width=1,
        shift=1,
        label_columns=['T (degC)'],
        training_set=training_set
    )

    compile_and_fit(conv_model, conv_window)

    LABEL_WIDTH = 24
    INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
    wide_conv_window = WindowGenerator(
        input_width=INPUT_WIDTH,
        label_width=LABEL_WIDTH,
        shift=1,
        label_columns=['T (degC)'],
        training_set=training_set
    )

    print("Wide conv window")
    print('Input shape:', wide_conv_window.example[0].shape)
    print('Labels shape:', wide_conv_window.example[1].shape)
    print('Output shape:', conv_model(wide_conv_window.example[0]).shape)

    wide_conv_window.plot(conv_model)

    return TrainingResult(
        validation_performance=conv_model.evaluate(conv_window.val),
        performance=conv_model.evaluate(conv_window.test, verbose=0)
    )


def test_multi_recurrent(
        training_set: TrainingSet
) -> TrainingResult:
    lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(32, return_sequences=True),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
    ])

    wide_window = WindowGenerator(
        input_width=24,
        label_width=24,
        shift=1,
        label_columns=['T (degC)'],
        training_set=training_set
    )

    print('Input shape:', wide_window.example[0].shape)
    print('Output shape:', lstm_model(wide_window.example[0]).shape)

    compile_and_fit(lstm_model, wide_window)

    wide_window.plot(lstm_model)

    return TrainingResult(
        validation_performance=lstm_model.evaluate(wide_window.val),
        performance=lstm_model.evaluate(wide_window.test, verbose=0)
    )


def plot_single_output(
        results: Dict[str, TrainingResult]
):
    plt.figure()
    x = np.arange(len(results))
    width = 0.3
    # metric_name = 'mean_absolute_error'
    metric_index = results.values()[-1].metrics_names.index('mean_absolute_error')
    val_mae = [v.validation_performance[metric_index] for v in results.values()]
    test_mae = [v.performance[metric_index] for v in results.values()]

    plt.ylabel('mean_absolute_error [T (degC), normalized]')
    plt.bar(x - 0.17, val_mae, width, label='Validation')
    plt.bar(x + 0.17, test_mae, width, label='Test')
    plt.xticks(
        ticks=x,
        labels=results.keys(),
        rotation=45
    )
    _ = plt.legend()


def run():
    data = get_data()
    data = clean_data(data)
    data = transform_data(data)

    training_set = prepare_sets(data)

    results = {
        'baseline': test_baseline(training_set=training_set),
        'linear': test_linear(training_set=training_set),
        'dense': test_dense(training_set=training_set),
        'multi-step-dense': test_multi_step_dense(training_set=training_set),
        'multi-step-conv': test_multi_step_conv_net(training_set=training_set),
        'multi-step-recurrent': test_multi_recurrent(training_set=training_set)
    }

    print(results)

    plot_single_output(results=results)


if __name__ == '__main__':
    run()
