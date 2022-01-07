from pathlib import Path
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt

from data_types.training_result import TrainingResult


def plot_single_output(
        results: Dict[str, TrainingResult],
        path_save: Path
) -> None:
    plt.figure()
    x = np.arange(len(results))
    width = 0.3

    val_mae = [v.validation_performance for v in results.values()]
    test_mae = [v.performance for v in results.values()]

    plt.ylabel('mean_absolute_error [T (degC), normalized]')
    plt.bar(x - 0.17, val_mae, width, label='Validation')
    plt.bar(x + 0.17, test_mae, width, label='Test')
    plt.xticks(
        ticks=x,
        labels=results.keys(),
        rotation=45
    )
    _ = plt.legend()

    plt.gcf().subplots_adjust(bottom=0.25)

    plt.savefig(path_save / "results_single.png")
    plt.close()


def plot_multi_output(
        results: Dict[str, TrainingResult],
        path_save: Path
) -> None:
    x = np.arange(len(results))
    width = 0.3

    val_mae = [v.validation_performance for v in results.values()]
    test_mae = [v.performance for v in results.values()]

    plt.bar(x - 0.17, val_mae, width, label='Validation')
    plt.bar(x + 0.17, test_mae, width, label='Test')
    plt.xticks(ticks=x, labels=results.keys(),
               rotation=45)
    plt.ylabel('MAE (average over all outputs)')
    _ = plt.legend()

    plt.gcf().subplots_adjust(bottom=0.25)

    plt.savefig(path_save / "results_multi_output.png")
    plt.close()


def plot_multi_output_multi_step(
        results: Dict[str, TrainingResult],
        path_save: Path
) -> None:
    x = np.arange(len(results))
    width = 0.3

    val_mae = [v.validation_performance for v in results.values()]
    test_mae = [v.performance for v in results.values()]

    plt.bar(x - 0.17, val_mae, width, label='Validation')
    plt.bar(x + 0.17, test_mae, width, label='Test')
    plt.xticks(ticks=x, labels=results.keys(),
               rotation=45)
    plt.ylabel(f'MAE (average over all times and outputs)')
    _ = plt.legend()

    plt.gcf().subplots_adjust(bottom=0.25)

    plt.savefig(path_save / "results_multi_output_multi_step.png")
    plt.close()
