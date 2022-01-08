from pathlib import Path
from typing import List

import tensorflow as tf
from matplotlib import pyplot as plt

from data_types.training_result import TrainingResult
from data_types.training_set import TrainingSet
from timeseries.baseline import Baseline
from timeseries.window_generator import WindowGenerator


def evaluate_baseline_multi_output(
        training_set: TrainingSet,
        label_columns: List[str],
        path_save: Path
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
        label_columns=label_columns,
        training_set=training_set
    )

    evaluation = baseline.evaluate(wide_window.val)

    metric_index = baseline.metrics_names.index('mean_absolute_error')

    res = TrainingResult(
        validation_performance=evaluation[metric_index],
        performance=baseline.evaluate(wide_window.test, verbose=0)[metric_index]
    )

    wide_window.plot(
        plot_col=label_columns[0],
        model=baseline,
        path_save=path_save / "multi_output_baseline.jpg"
    )

    return res
