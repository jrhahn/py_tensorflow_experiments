import tensorflow as tf

from data_types.training_result import TrainingResult
from data_types.training_set import TrainingSet
from timeseries.baseline import Baseline
from timeseries.window_generator import WindowGenerator


def evaluate_baseline_multi_output(
        training_set: TrainingSet
) -> TrainingResult:
    baseline = Baseline()
    baseline.compile(
        loss=tf.losses.MeanSquaredError(),
        metrics=[tf.metrics.MeanAbsoluteError()]
    )

    wide_window = WindowGenerator(
        input_width=24,
        label_width=24,
        shift=1,
        label_columns=['T (degC)'],
        training_set=training_set
    )

    evaluation = baseline.evaluate(wide_window.val)

    metric_index = baseline.metrics_names.index('mean_absolute_error')

    return TrainingResult(
        validation_performance=evaluation[metric_index],
        performance=baseline.evaluate(wide_window.test, verbose=0)[metric_index]
    )