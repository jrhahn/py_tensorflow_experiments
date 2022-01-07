from typing import List

import tensorflow as tf

from data_types.training_result import TrainingResult
from data_types.training_set import TrainingSet
from timeseries.build import compile_and_fit
from timeseries.constants import CONV_WIDTH
from timeseries.window_generator import WindowGenerator


def evaluate_multi_step_dense(
        training_set: TrainingSet,
        label_columns: List[str] = ['T (degC)']
) -> TrainingResult:
    conv_window = WindowGenerator(
        input_width=CONV_WIDTH,
        label_width=1,
        shift=1,
        label_columns=label_columns,
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

    metric_index = multi_step_dense.metrics_names.index('mean_absolute_error')

    return TrainingResult(
        validation_performance=multi_step_dense.evaluate(conv_window.val)[metric_index],
        performance=multi_step_dense.evaluate(conv_window.test, verbose=0)[metric_index]
    )
