import pandas as pd


def stdmean_scaler(x_set: pd.Dataframe) -> pd.DataFrame:
    x_means = pd.DataFrame
    x_std = pd.DataFrame
    for col in x_set.keys():
        x_means[col] = x_set[col].mean()
        x_std[col] = x_set[col].std()
        x_set[col] = (x_set[col] - x_means[col]) / x_std[col]

    return x_set
