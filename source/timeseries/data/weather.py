import os
from pathlib import Path

import pandas as pd
import tensorflow as tf

from data_types.training_set import TrainingSet


def get_data(path_save: Path) -> pd.DataFrame:
    zip_path = tf.keras.utils.get_file(
        origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
        fname='jena_climate_2009_2016.csv.zip',
        extract=True)
    csv_path, _ = os.path.splitext(zip_path)

    """
    ,
    Date Time,
    p (mbar),
    T (degC),
    Tpot (K),
    Tdew (degC),
    rh (%),
    VPmax (mbar),
    VPact (mbar),
    VPdef (mbar),
    sh (g/kg),
    H2OC (mmol/mol),
    rho (g/m**3),
    wv (m/s),
    max. wv (m/s),
    wd (deg)
    """

    df = pd.read_csv(csv_path)
    df.to_csv(path_save / "weather.csv")

    # Slice [start:stop:step], starting from index 5 take every 6th record.
    df = df[5::6]

    return df


def prepare_sets(df: pd.DataFrame) -> TrainingSet:
    n = len(df)
    train_df = df[0:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.9)]
    test_df = df[int(n * 0.9):]

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    df_std = (df - train_mean) / train_std
    df_std = df_std.melt(var_name='Column', value_name='Normalized')
    # plt.figure(figsize=(12, 6))
    # ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
    # _ = ax.set_xticklabels(df.keys(), rotation=90)

    return TrainingSet(
        training=train_df,
        test=test_df,
        validation=val_df,
        num_features=df.shape[1]
    )
