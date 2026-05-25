from pandas import DataFrame
import pandas as pd

def get_outliers(df: DataFrame, column: str) -> DataFrame:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    outliers = df[
        (df[column] < Q1 - 1.5 * IQR) |
        (df[column] > Q3 + 1.5 * IQR)
    ]

    return outliers