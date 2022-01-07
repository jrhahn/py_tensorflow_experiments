from typing import List

import tensorflow as tf

from data_types.training_result import TrainingResult
from data_types.training_set import TrainingSet
from timeseries.build import compile_and_fit
from timeseries.constants import CONV_WIDTH
from timeseries.window_generator import WindowGenerator


def evaluate_multi_step_conv_net(
        training_set: TrainingSet,
        label_columns: List[str] = ['T (degC)']
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
        label_columns=label_columns,
        training_set=training_set
    )

    compile_and_fit(conv_model, conv_window)

    LABEL_WIDTH = 24
    INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
    wide_conv_window = WindowGenerator(
        input_width=INPUT_WIDTH,
        label_width=LABEL_WIDTH,
        shift=1,
        label_columns=label_columns,
        training_set=training_set
    )

    print("Wide conv window")
    print('Input shape:', wide_conv_window.example[0].shape)
    print('Labels shape:', wide_conv_window.example[1].shape)
    print('Output shape:', conv_model(wide_conv_window.example[0]).shape)

    wide_conv_window.plot(conv_model)

    metric_index = conv_model.metrics_names.index('mean_absolute_error')

    return TrainingResult(
        validation_performance=conv_model.evaluate(conv_window.val)[metric_index],
        performance=conv_model.evaluate(conv_window.test, verbose=0)[metric_index]
    )
