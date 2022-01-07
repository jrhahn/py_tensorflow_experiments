import tensorflow as tf

from data_types.training_result import TrainingResult
from data_types.training_set import TrainingSet
from timeseries.baseline import Baseline
from timeseries.window_generator import WindowGenerator


def evaluate_baseline(
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

    evaluation = baseline.evaluate(single_step_window.val)

    metric_index = baseline.metrics_names.index('mean_absolute_error')

    result = TrainingResult(
        validation_performance=evaluation[metric_index],
        performance=baseline.evaluate(single_step_window.test, verbose=0)[metric_index]
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