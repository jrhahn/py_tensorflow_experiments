from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_data(
        path_data: Path,
        path_save: Path,
        filename: str
) -> pd.DataFrame:
    df = pd.read_csv(path_data / filename, skiprows=1)
    # df['date'] = pd.to_datetime(df['date'])
    # df.set_index('date', inplace=True)
    df.dropna(inplace=True)

    # df['close'].dropna().plot()
    # plt.savefig(path_save / 'data.png')

    day = 24 * 60 * 60
    year = (365.2425) * day

    timestamp_s = df['unix'] // 1000

    # df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    # df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    df.drop(['unix', 'date', 'symbol'], axis=1, inplace=True)

    return df
