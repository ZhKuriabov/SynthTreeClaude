from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm


def _add_constant(X: np.ndarray) -> np.ndarray:
    return sm.add_constant(np.asarray(X, dtype=float), has_constant="add")


def _nondegenerate_columns(X: np.ndarray) -> np.ndarray:
    """Return a boolean mask of columns that are not constant and not
    perfectly collinear with an earlier column.  The first column (intercept)
    is always kept."""
    n, p = X.shape
    keep = np.ones(p, dtype=bool)
    # Drop constant columns (skip col 0 which is the intercept)
    for j in range(1, p):
        if np.ptp(X[:, j]) == 0.0:
            keep[j] = False
    # QR-based rank detection on the kept columns
    kept_idx = np.where(keep)[0]
    if kept_idx.size <= 1:
        return keep
    Q, R = np.linalg.qr(X[:, kept_idx])
    diag_abs = np.abs(np.diag(R))
    tol = max(n, kept_idx.size) * np.finfo(float).eps * (diag_abs[0] if diag_abs[0] > 0 else 1.0)
    for i, j in enumerate(kept_idx):
        if diag_abs[i] < tol:
            keep[j] = False
    return keep


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
        self._col_mask = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.shape[0] == 0:
            self.model_ = ConstantModel(classification=False, value=0.0)
            self._col_mask = None
            return self
        if np.allclose(y, y[0]):
            self.model_ = ConstantModel(classification=False, value=float(y[0]))
            self._col_mask = None
            return self

        exog = _add_constant(X)
        col_mask = _nondegenerate_columns(exog)
        self._col_mask = col_mask
        self.model_ = sm.OLS(y, exog[:, col_mask]).fit_regularized(
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
        exog = _add_constant(X)
        return np.asarray(self.model_.predict(exog[:, self._col_mask]), dtype=float)

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
        self._col_mask = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.shape[0] == 0:
            self.model_ = ConstantModel(classification=True, value=0.5)
            self._col_mask = None
            return self
        if np.unique(y).size == 1:
            self.model_ = ConstantModel(classification=True, value=float(y[0]))
            self._col_mask = None
            return self

        exog = _add_constant(X)
        col_mask = _nondegenerate_columns(exog)
        self._col_mask = col_mask
        self.model_ = sm.Logit(y, exog[:, col_mask]).fit_regularized(
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
        exog = _add_constant(X)
        return np.asarray(self.model_.predict(exog[:, self._col_mask]), dtype=float)

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
