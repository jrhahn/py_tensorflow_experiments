import tensorflow as tf

from data_types.training_result import TrainingResult
from data_types.training_set import TrainingSet
from timeseries.build import compile_and_fit
from timeseries.constants import CONV_WIDTH, OUT_STEPS
from timeseries.window_generator import WindowGenerator


def evaluate_conv_multi_output_multi_step(
        training_set: TrainingSet
) -> TrainingResult:
    multi_conv_model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
        tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
        # Shape => [batch, 1, conv_units]
        tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
        # Shape => [batch, 1,  out_steps*features]
        tf.keras.layers.Dense(OUT_STEPS * training_set.num_features,
                              kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, training_set.num_features])
    ])

    multi_window = WindowGenerator(
        input_width=24,
        label_width=OUT_STEPS,
        shift=OUT_STEPS,
        training_set=training_set
    )

    compile_and_fit(multi_conv_model, multi_window)

    metric_index = multi_conv_model.metrics_names.index('mean_absolute_error')

    return TrainingResult(
        validation_performance=multi_conv_model.evaluate(multi_window.val)[metric_index],
        performance=multi_conv_model.evaluate(multi_window.test, verbose=0)[metric_index]
    )

    # multi_window.plot(multi_conv_model)
