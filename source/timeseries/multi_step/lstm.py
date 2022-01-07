import tensorflow as tf

from data_types.training_result import TrainingResult
from data_types.training_set import TrainingSet
from timeseries.build import compile_and_fit
from timeseries.window_generator import WindowGenerator


def evaluate_multi_step_recurrent(
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

    metric_index = lstm_model.metrics_names.index('mean_absolute_error')

    return TrainingResult(
        validation_performance=lstm_model.evaluate(wide_window.val)[metric_index],
        performance=lstm_model.evaluate(wide_window.test, verbose=0)[metric_index]
    )