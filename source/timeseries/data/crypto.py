from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_data(
        path_data: Path,
        path_save: Path
) -> pd.DataFrame:
    df = pd.read_csv(path_data / "Binance_ADAUSDT_d.csv", skiprows=1)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # df['close'].dropna().plot()
    # plt.savefig(path_save / 'data.png')

    return df
