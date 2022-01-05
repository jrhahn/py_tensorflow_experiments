from dataclasses import dataclass
import pandas as pd


@dataclass
class TrainingSet:
    training: pd.DataFrame
    test: pd.DataFrame
    validation: pd.DataFrame
