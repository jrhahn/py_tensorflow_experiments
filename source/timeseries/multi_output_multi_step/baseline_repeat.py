from pathlib import Path
from typing import List

import tensorflow as tf

from data_types.training_result import TrainingResult
from data_types.training_set import TrainingSet
from timeseries.constants import OUT_STEPS
from timeseries.multi_output_multi_step.repeat_baseline import RepeatBaseline
from timeseries.window_generator import WindowGenerator


def evaluate_repeat_baseline_multi_output_multi_step(
        training_set: TrainingSet,
        label_columns: List[str],
        path_save: Path
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

    evaluation = repeat_baseline.evaluate(multi_window.val)

    metric_index = repeat_baseline.metrics_names.index('mean_absolute_error')

    res = TrainingResult(
        validation_performance=evaluation[metric_index],
        performance=repeat_baseline.evaluate(multi_window.test, verbose=0)[metric_index]
    )

    multi_window.plot(
        plot_col=label_columns[0],
        model=repeat_baseline,
        path_save=path_save / "multi_output_multi_step_baseline_repeat.jpg"
    )

    return res
