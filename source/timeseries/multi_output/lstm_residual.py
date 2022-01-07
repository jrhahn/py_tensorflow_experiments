from typing import List

import tensorflow as tf

from data_types.training_result import TrainingResult
from data_types.training_set import TrainingSet
from timeseries.build import compile_and_fit
from timeseries.multi_output.residual_wrapper import ResidualWrapper
from timeseries.window_generator import WindowGenerator


def evaluate_residual_lstm_multi_output(
        training_set: TrainingSet,
        label_columns: List[str] = ['T (degC)']
) -> TrainingResult:
    residual_lstm = ResidualWrapper(
        tf.keras.Sequential([
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.Dense(
                training_set.num_features,
                # The predicted deltas should start small.
                # Therefore, initialize the output layer with zeros.
                kernel_initializer=tf.initializers.zeros())
        ]))

    wide_window = WindowGenerator(
        input_width=24,
        label_width=24,
        shift=1,
        label_columns=label_columns,
        training_set=training_set
    )

    compile_and_fit(residual_lstm, wide_window)

    metric_index = residual_lstm.metrics_names.index('mean_absolute_error')

    return TrainingResult(
        validation_performance=residual_lstm.evaluate(wide_window.val)[metric_index],
        performance=residual_lstm.evaluate(wide_window.test, verbose=0)[metric_index]
    )