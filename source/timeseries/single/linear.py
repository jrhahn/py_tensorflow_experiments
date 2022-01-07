from typing import List

import tensorflow as tf

from data_types.training_result import TrainingResult
from data_types.training_set import TrainingSet
from timeseries.build import compile_and_fit
from timeseries.window_generator import WindowGenerator


def evaluate_linear(
        training_set: TrainingSet,
        label_columns: List[str]
) -> TrainingResult:
    linear = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1)
    ])

    single_step_window = WindowGenerator(
        input_width=1,
        label_width=1,
        shift=1,
        training_set=training_set,
        label_columns=label_columns
    )

    print('Input shape:', single_step_window.example[0].shape)
    print('Output shape:', linear(single_step_window.example[0]).shape)

    compile_and_fit(linear, single_step_window)

    wide_window = WindowGenerator(
        input_width=24,
        label_width=24,
        shift=1,
        label_columns=label_columns,
        training_set=training_set
    )
    wide_window.plot(
        model=linear,
        plot_col=label_columns[0]
    )

    metric_index = linear.metrics_names.index('mean_absolute_error')

    return TrainingResult(
        performance=linear.evaluate(single_step_window.test, verbose=0)[metric_index],
        validation_performance=linear.evaluate(single_step_window.val)[metric_index]
    )
