import tensorflow as tf

from data_types.training_result import TrainingResult
from data_types.training_set import TrainingSet
from timeseries.constants import OUT_STEPS
from timeseries.multi_output_multi_step.repeat_baseline import RepeatBaseline
from timeseries.window_generator import WindowGenerator


def evaluate_repeat_baseline_multi_output_multi_step(
        training_set: TrainingSet
) -> TrainingResult:
    repeat_baseline = RepeatBaseline()
    repeat_baseline.compile(
        loss=tf.losses.MeanSquaredError(),
        metrics=[tf.metrics.MeanAbsoluteError()]
    )

    multi_window = WindowGenerator(
        input_width=24,
        label_width=OUT_STEPS,
        shift=OUT_STEPS,
        training_set=training_set
    )

    return TrainingResult(
        validation_performance=repeat_baseline.evaluate(multi_window.val),
        performance=repeat_baseline.evaluate(multi_window.test, verbose=0)
    )
    # multi_val_performance['Repeat'] =
    # multi_performance['Repeat'] =
    # multi_window.plot(repeat_baseline)