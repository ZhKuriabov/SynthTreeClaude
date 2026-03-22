from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm


def _add_constant(X: np.ndarray) -> np.ndarray:
    return sm.add_constant(np.asarray(X, dtype=float), has_constant="add")


@dataclass
class ConstantModel:
    classification: bool
    value: float

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        return np.repeat(float(self.value), len(X))

    def predict_label(self, X: np.ndarray) -> np.ndarray:
        if self.classification:
            return (self.predict_score(X) >= 0.5).astype(int)
        return self.predict_score(X)

    def complexity(self) -> int:
        return 0


class StatsmodelsLinearModel:
    def __init__(self, alpha: float = 0.1, max_iter: int = 1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.model_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.shape[0] == 0:
            self.model_ = ConstantModel(classification=False, value=0.0)
            return self
        if np.allclose(y, y[0]):
            self.model_ = ConstantModel(classification=False, value=float(y[0]))
            return self

        exog = _add_constant(X)
        self.model_ = sm.OLS(y, exog).fit_regularized(
            method="elastic_net",
            alpha=self.alpha,
            L1_wt=1.0,
            maxiter=self.max_iter,
        )
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if isinstance(self.model_, ConstantModel):
            return self.model_.predict_score(X)
        return np.asarray(self.model_.predict(_add_constant(X)), dtype=float)

    def predict_label(self, X: np.ndarray) -> np.ndarray:
        return self.predict_score(X)

    def complexity(self) -> int:
        if isinstance(self.model_, ConstantModel):
            return 0
        params = np.asarray(self.model_.params, dtype=float)
        return int(np.count_nonzero(np.abs(params[1:]) > 1e-8))


class StatsmodelsBinaryModel:
    def __init__(self, alpha: float = 0.1, max_iter: int = 1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.model_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.shape[0] == 0:
            self.model_ = ConstantModel(classification=True, value=0.5)
            return self
        if np.unique(y).size == 1:
            self.model_ = ConstantModel(classification=True, value=float(y[0]))
            return self

        exog = _add_constant(X)
        self.model_ = sm.Logit(y, exog).fit_regularized(
            method="l1",
            alpha=self.alpha,
            maxiter=self.max_iter,
            disp=0,
        )
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if isinstance(self.model_, ConstantModel):
            return self.model_.predict_score(X)
        return np.asarray(self.model_.predict(_add_constant(X)), dtype=float)

    def predict_label(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_score(X) >= 0.5).astype(int)

    def complexity(self) -> int:
        if isinstance(self.model_, ConstantModel):
            return 0
        params = np.asarray(self.model_.params, dtype=float)
        return int(np.count_nonzero(np.abs(params[1:]) > 1e-8))


def make_statsmodels_model(classification: bool, alpha: float, max_iter: int):
    if classification:
        return StatsmodelsBinaryModel(alpha=alpha, max_iter=max_iter)
    return StatsmodelsLinearModel(alpha=alpha, max_iter=max_iter)
