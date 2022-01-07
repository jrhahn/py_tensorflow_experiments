import tensorflow as tf

from data_types.training_result import TrainingResult
from data_types.training_set import TrainingSet
from timeseries.build import compile_and_fit
from timeseries.constants import OUT_STEPS
from timeseries.window_generator import WindowGenerator


def evaluate_linear_multi_output_multi_step(
        training_set: TrainingSet
) -> TrainingResult:
    multi_linear_model = tf.keras.Sequential(
        [
            # Take the last time-step.
            # Shape [batch, time, features] => [batch, 1, features]
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
            # Shape => [batch, 1, out_steps*features]
            tf.keras.layers.Dense(
                OUT_STEPS * training_set.num_features,
                kernel_initializer=tf.initializers.zeros()
            ),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([OUT_STEPS, training_set.num_features])
        ]
    )

    multi_window = WindowGenerator(
        input_width=24,
        label_width=OUT_STEPS,
        shift=OUT_STEPS,
        training_set=training_set
    )

    compile_and_fit(multi_linear_model, multi_window)

    metric_index = multi_linear_model.metrics_names.index('mean_absolute_error')

    return TrainingResult(
        validation_performance=multi_linear_model.evaluate(multi_window.val)[metric_index],
        performance=multi_linear_model.evaluate(multi_window.test, verbose=0)[metric_index]
    )
    # multi_val_performance['Linear'] =
    # multi_performance['Linear'] =
    # multi_window.plot(multi_linear_model)
