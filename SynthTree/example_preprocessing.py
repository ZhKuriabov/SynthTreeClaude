from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

# -----------------------------------------------------------------------------
# Helper metric for regression
# -----------------------------------------------------------------------------
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# -----------------------------------------------------------------------------
# Main loader / pre-processor
# -----------------------------------------------------------------------------

def prep_data_with_feature_names(
    data_file: str,
    random_state: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return train/test split and processed feature names for the requested dataset label."""

    # default test split fraction
    test_frac = 0.2

    if data_file == "SKCM":
        df = pd.read_csv("./data/TCGA_skcm.csv")
        data = df.iloc[:, 1:]
        outcome_col = "OS_STATUS"

        feature_names = [c for c in data.columns if c != outcome_col and len(np.unique(data[c])) > 1]
        Xo = data[feature_names].values
        y = data[outcome_col].values

        dfX = pd.DataFrame(Xo, columns=feature_names)
        categorical = [col for col in dfX.columns if dfX[col].dtype == object or len(np.unique(dfX[col])) < 5]
        continuous  = [col for col in dfX.columns if col not in categorical]

        X_cont = dfX[continuous].astype(float)
        X_cont = (X_cont - X_cont.mean()) / (X_cont.std() + 1e-8)
        X_cat  = pd.get_dummies(dfX[categorical], drop_first=False)
        X_df   = pd.concat([X_cont, X_cat], axis=1)

        feature_names = list(X_df.columns)
        X_train, X_test, y_train, y_test = train_test_split(
            X_df.values,
            y,
            test_size=test_frac,
            random_state=random_state,
            shuffle=True
        )

    elif data_file == "Servo":
        df  = pd.read_csv("./data/servo.csv")
        data = df.iloc[:, :-1]
        y    = df.iloc[:,  -1].values

        new_df = pd.get_dummies(data.loc[:, ["Motor", "Screw"]], dtype=float)
        old_df = data.loc[:, ["Vgain", "Pgain"]].copy()
        for column in old_df.columns:
            old_df[column] = (old_df[column] - old_df[column].min()) / (
                old_df[column].max() - old_df[column].min()
            )
        new_df["Pgain"], new_df["Vgain"] = old_df["Pgain"], old_df["Vgain"]
        X = new_df.values

        feature_names = list(new_df.columns)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_frac,
            random_state=random_state,
            shuffle=True
        )

    elif data_file == "Road Safety":
        df = pd.read_csv("data/road_safety_dataset.csv", header=None)

        # target is col 32 (1 → 0, 2 → 1)
        y = df[32].replace({1: 0, 2: 1}).values
        X_df = df.drop(columns=32)

        # min-max scale every feature
        X = ((X_df - X_df.min()) / (X_df.max() - X_df.min() + 1e-9)).values

        feature_names = [f"V{idx}" for idx in range(X.shape[1])]
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_frac,
            random_state=random_state,
            shuffle=True
        )

    elif data_file == "Upselling":
        df = pd.read_csv("data/KDDCup09_upselling_dataset.csv", header=None)
        df[41].replace("cJvF", 0, inplace=True)
        df[41].replace("UYBR", 1, inplace=True)

        for column in df.columns:
            df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

        X_df, y_df = df.iloc[:, :-1].values, df.iloc[:, -1].values

        feature_names = [f"V{idx}" for idx in range(X_df.shape[1])]
        X_train, X_test, y_train, y_test = train_test_split(
            X_df,
            y_df,
            test_size=test_frac,
            random_state=random_state,
            shuffle=True
        )

    elif data_file == "Compas":
        df = pd.read_csv("data/compass_dataset.csv", header=None)
        df = df.drop(columns=13)
        X_df = df.drop(columns=df.columns[-1])
        X = ((X_df - X_df.min()) / (X_df.max() - X_df.min() + 1e-9)).values
        y = df.iloc[:, -1].astype(int).values

        feature_names = [f"V{idx}" for idx in range(X.shape[1])]
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_frac,
            random_state=random_state,
            shuffle=True
        )

    elif data_file == "Bike Sharing":
        path = "./data/"
        df   = pd.read_csv(path + "hour.csv.gz", compression="gzip")

        outcome_header = "cnt"
        used_headers   = list(df)
        for drop_col in ["instant", "dteday", "casual", "registered"]:
            used_headers.remove(drop_col)
        data = df[used_headers]

        def add_dummy(col: str, data_: pd.DataFrame):
            dummies = pd.get_dummies(data_[col], prefix=col, dtype=float)
            data_   = pd.concat([data_.drop(columns=col), dummies], axis=1)
            return data_

        for col in ["season", "weathersit"]:
            data = add_dummy(col, data)

        y = data[outcome_header].values
        X = data.drop(columns=outcome_header).values

        feature_names = list(data.drop(columns=outcome_header).columns)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_frac,
            random_state=random_state,
            shuffle=True
        )

    elif data_file == "Abalone":
        df = pd.read_csv("./data/abalone.csv")
        X_df = pd.get_dummies(df.iloc[:, :-1], dtype=float)
        X = ((X_df - X_df.min()) / (X_df.max() - X_df.min() + 1e-9)).values
        y = df.iloc[:, -1].values

        feature_names = list(X_df.columns)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_frac,
            random_state=random_state,
            shuffle=True
        )

    elif data_file == "Cal Housing":
        from sklearn.datasets import fetch_california_housing
        cal = fetch_california_housing()
        Xo  = cal["data"].copy()
        y   = cal["target"].copy()

        Xo[:, 2] = Xo[:, 2] - Xo[:, 3]
        X = (Xo - Xo.min(axis=0)) / (Xo.max(axis=0) - Xo.min(axis=0) + 1e-9)

        feature_names = list(cal["feature_names"])
        feature_names[2] = "AveRoomsMinusAveBedrms"
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_frac,
            random_state=random_state,
            shuffle=True
        )

    else:
        raise ValueError(f"Unknown data_file label: {data_file}")

    return X_train, y_train, X_test, y_test, feature_names


def prep_data(data_file: str, random_state: int | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return train/test split for the requested *data_file* label with reproducible RNG."""
    X_train, y_train, X_test, y_test, _ = prep_data_with_feature_names(
        data_file=data_file,
        random_state=random_state,
    )
    return X_train, y_train, X_test, y_test
