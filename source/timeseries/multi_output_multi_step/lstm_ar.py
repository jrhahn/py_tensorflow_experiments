from pathlib import Path
from typing import List

from data_types.training_result import TrainingResult
from data_types.training_set import TrainingSet
from timeseries.build import compile_and_fit
from timeseries.constants import OUT_STEPS
from timeseries.multi_output_multi_step.feedback import FeedBack
from timeseries.window_generator import WindowGenerator


def evaluate_ar_lstm_multi_output_multi_step(
        training_set: TrainingSet,
        label_columns: List[str],
        path_save: Path
) -> TrainingResult:
    feedback_model = FeedBack(
        units=32,
        out_steps=OUT_STEPS,
        num_features=training_set.num_features
    )

    multi_window = WindowGenerator(
        input_width=24,
        label_width=OUT_STEPS,
        shift=OUT_STEPS,
        training_set=training_set
    )

    feedback_model.warmup(multi_window.example[0])

    compile_and_fit(feedback_model, multi_window)

    metric_index = feedback_model.metrics_names.index('mean_absolute_error')

    res = TrainingResult(
        validation_performance=feedback_model.evaluate(multi_window.val)[metric_index],
        performance=feedback_model.evaluate(multi_window.test, verbose=0)[metric_index]
    )

    multi_window.plot(
        plot_col=label_columns[0],
        model=feedback_model,
        path_save=path_save / "multi_output_multi_step_lstm_ar.jpg"
    )

    return res
