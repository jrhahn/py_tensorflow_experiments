import pandas as pd

from data_types.training_set import TrainingSet


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

    # df_std = (df - train_mean) / train_std
    # df_std = df_std.melt(var_name='Column', value_name='Normalized')
    # plt.figure(figsize=(12, 6))
    # ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
    # _ = ax.set_xticklabels(df.keys(), rotation=90)

    return TrainingSet(
        training=train_df,
        test=test_df,
        validation=val_df,
        num_features=df.shape[1]
    )
