from pathlib import Path

from repository import RepositoryInfo
from timeseries.data import weather, crypto
from timeseries.data.common import prepare_sets
from timeseries.data.pre_processing import clean_data, transform_data
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
    # # weather data
    # dataset = "weather"
    # dataset = "ADAUSDT"
    dataset = "BNBUSDT"
    # dataset = "BTCUSDT"
    # dataset = "DOTUSDT"
    # dataset = "ETHUSDT"
    # dataset = "XLMUSDT"

    repo_info = RepositoryInfo(sub_folder_save=str(Path('plots') / dataset))

    if dataset == "weather":
        data = weather.get_data(path_save=RepositoryInfo(sub_folder_save='data').path_save)
        data = clean_data(data)
        data = transform_data(data)
        target_columns = ['T (degC)']
    else:
        data = crypto.get_data(
            path_data=repo_info.path_tmp / 'data' / 'crypto_download_data',
            path_save=repo_info.path_save,
            filename=f"Binance_{dataset}_d.csv"
        )
        target_columns = ['close']

    training_set = prepare_sets(data)

    results_single = {
        'baseline': evaluate_baseline(
            training_set=training_set,
            label_columns=target_columns,
            path_save=repo_info.path_save
        ),
        'linear': evaluate_linear(
            training_set=training_set,
            label_columns=target_columns,
            path_save=repo_info.path_save
        ),
        'dense': evaluate_dense(
            training_set=training_set,
            label_columns=target_columns,
            path_save=repo_info.path_save
        ),
        'multi-step-dense': evaluate_multi_step_dense(
            training_set=training_set,
            label_columns=target_columns,
            path_save=repo_info.path_save
        ),
        'multi-step-conv': evaluate_multi_step_conv_net(
            training_set=training_set,
            label_columns=target_columns,
            path_save=repo_info.path_save
        ),
        'multi-step-recurrent': evaluate_multi_step_recurrent(
            training_set=training_set,
            label_columns=target_columns,
            path_save=repo_info.path_save
        )
    }

    plot_single_output(
        results=results_single,
        path_save=repo_info.path_save
    )

    results_multi_output = {
        'baseline': evaluate_baseline_multi_output(
            training_set=training_set,
            label_columns=target_columns,
            path_save=repo_info.path_save
        ),
        'dense': evaluate_dense_multi_output(
            training_set=training_set,
            label_columns=target_columns,
            path_save=repo_info.path_save
        ),
        'lstm': evaluate_lstm_multi_output(
            training_set=training_set,
            label_columns=target_columns,
            path_save=repo_info.path_save
        ),
        'residual_lstm': evaluate_residual_lstm_multi_output(
            training_set=training_set,
            label_columns=target_columns,
            path_save=repo_info.path_save
        )
    }

    plot_multi_output(
        results=results_multi_output,
        path_save=repo_info.path_save
    )

    results_multi_output_multi_step = {
        'baseline': evaluate_baseline_multi_output_multi_step(
            training_set=training_set,
            label_columns=target_columns,
            path_save=repo_info.path_save
        ),
        'repeat_baseline': evaluate_repeat_baseline_multi_output_multi_step(
            training_set=training_set,
            label_columns=target_columns,
            path_save=repo_info.path_save
        ),
        'linear': evaluate_linear_multi_output_multi_step(
            training_set=training_set,
            label_columns=target_columns,
            path_save=repo_info.path_save
        ),
        'dense': evaluate_dense_multi_output_multi_step(
            training_set=training_set,
            label_columns=target_columns,
            path_save=repo_info.path_save
        ),
        'conv': evaluate_conv_multi_output_multi_step(
            training_set=training_set,
            label_columns=target_columns,
            path_save=repo_info.path_save
        ),
        'lstm': evaluate_lstm_multi_output_multi_step(
            training_set=training_set,
            label_columns=target_columns,
            path_save=repo_info.path_save
        ),
        'ar_lstm': evaluate_ar_lstm_multi_output_multi_step(
            training_set=training_set,
            label_columns=target_columns,
            path_save=repo_info.path_save
        )
    }

    plot_multi_output_multi_step(
        results=results_multi_output_multi_step,
        path_save=repo_info.path_save
    )


if __name__ == '__main__':
    run()
