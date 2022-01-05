import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from base_line import Baseline
from data_types.training_result import TrainingResult
from data_types.training_set import TrainingSet
from window_generator import WindowGenerator


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
    ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
    _ = ax.set_xticklabels(df.keys(), rotation=90)

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


def run():
    data = get_data()
    data = clean_data(data)
    data = transform_data(data)

    training_set = prepare_sets(data)

    test_baseline(training_set=training_set)


if __name__ == '__main__':
    run()
