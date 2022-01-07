from repository import RepositoryInfo
from timeseries.data.pre_processing import clean_data, transform_data
from timeseries.data.weather import get_data, prepare_sets
from timeseries.multi_output.baseline import evaluate_baseline_multi_output
from timeseries.multi_output.dense import evaluate_dense_multi_output
from timeseries.multi_output.lstm import evaluate_lstm_multi_output
from timeseries.multi_output.lstm_residual import evaluate_residual_lstm_multi_output
from timeseries.multi_output_multi_step.baseline import evaluate_baseline_multi_output_multi_step
from timeseries.multi_output_multi_step.baseline_repeat import evaluate_repeat_baseline_multi_output_multi_step
from timeseries.multi_output_multi_step.conv import evaluate_conv_multi_output_multi_step
from timeseries.multi_output_multi_step.dense import evaluate_dense_multi_output_multi_step
from timeseries.multi_output_multi_step.linear import evaluate_linear_multi_output_multi_step
from timeseries.multi_output_multi_step.lstm import evaluate_lstm_multi_output_multi_step
from timeseries.multi_output_multi_step.lstm_ar import evaluate_ar_lstm_multi_output_multi_step
from timeseries.multi_step.conv import evaluate_multi_step_conv_net
from timeseries.multi_step.dense import evaluate_multi_step_dense
from timeseries.multi_step.lstm import evaluate_multi_step_recurrent
from timeseries.plotting import plot_single_output, plot_multi_output, plot_multi_output_multi_step
from timeseries.single.baseline import evaluate_baseline
from timeseries.single.dense import evaluate_dense
from timeseries.single.linear import evaluate_linear


def run():
    data = get_data()
    data = clean_data(data)
    data = transform_data(data)

    repo_info = RepositoryInfo(sub_folder_save='plots')

    training_set = prepare_sets(data)

    results_single = {
        'baseline': evaluate_baseline(training_set=training_set),
        'linear': evaluate_linear(training_set=training_set),
        'dense': evaluate_dense(training_set=training_set),
        'multi-step-dense': evaluate_multi_step_dense(training_set=training_set),
        'multi-step-conv': evaluate_multi_step_conv_net(training_set=training_set),
        'multi-step-recurrent': evaluate_multi_step_recurrent(training_set=training_set)
    }

    print(results_single)
    plot_single_output(
        results=results_single,
        path_save=repo_info.path_save
    )

    results_multi_output = {
        'baseline_multi_output': evaluate_baseline_multi_output(training_set=training_set),
        'dense_multi_output': evaluate_dense_multi_output(training_set=training_set),
        'lstm_multi_output': evaluate_lstm_multi_output(training_set=training_set),
        'residual_lstm_multi_output': evaluate_residual_lstm_multi_output(training_set=training_set)
    }

    plot_multi_output(
        results=results_multi_output,
        path_save=repo_info.path_save
    )

    results_multi_output_multi_step = {
        'baseline_multi_output_multi_step': evaluate_baseline_multi_output_multi_step(
            training_set=training_set
        ),
        'repeat_baseline_multi_output_multi_step': evaluate_repeat_baseline_multi_output_multi_step(
            training_set=training_set
        ),
        'linear_multi_output_multi_step': evaluate_linear_multi_output_multi_step(
            training_set=training_set
        ),
        'dense_multi_output_multi_step': evaluate_dense_multi_output_multi_step(
            training_set=training_set
        ),
        'conv_multi_output_multi_step': evaluate_conv_multi_output_multi_step(
            training_set=training_set
        ),
        'lstm_multi_output_multi_step': evaluate_lstm_multi_output_multi_step(
            training_set=training_set
        ),
        'ar_lstm_multi_output_multi_step': evaluate_ar_lstm_multi_output_multi_step(
            training_set=training_set
        )
    }

    plot_multi_output_multi_step(
        results=results_multi_output_multi_step,
        path_save=repo_info.path_save
    )


if __name__ == '__main__':
    run()
