import pandas as pd
import numpy as np
from scipy.special import boxcox1p

def load_and_prepare(filepath, target_col='charges', transform=None):
    df = pd.read_csv(filepath)

    # Преобразуем бинарные строки
    df["sex"] = df["sex"].map({"male": 1, "female": 0})
    df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})

    # One-hot encoding 
    df = pd.get_dummies(df, columns=["region"], drop_first=True)

    # Преобразование целевой переменной
    if transform == "log":
        df[target_col] = np.log1p(df[target_col])
    elif transform == "box":
        df[target_col] = boxcox1p(df[target_col], 0.0435)
    elif transform is not None:
        raise ValueError("transform must be 'log', 'box', or None")

    return df
