import pandas as pd
import os

from src.models.pipeline import COLS, TARGET, DROP_COLS

def load_df(path = "data/interim/featured_12_02.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed data not found at: {path}")
    df = pd.read_csv(path)
    return df

def build_xy(df):
    return df[COLS].copy(), df[TARGET].values.ravel()

def load_data(years, path="data/interim/featured_12_02.csv"):
    df = load_df(path)

    df = df[df["season"].isin(years)].copy()
    df = df[(df['down'] == 3) | (df['down'] == 4)]

    X, y = build_xy(df)
    X = X.drop(columns = DROP_COLS)

    return X, y