import os
from pathlib import Path

import pandas as pd
import tensorflow as tf


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


