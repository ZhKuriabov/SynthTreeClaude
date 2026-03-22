from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

from .metrics import score_predictions, validation_objective
from .models import make_statsmodels_model


@dataclass
class CellModel:
    centroid: np.ndarray
    model: object
    X_cell: np.ndarray
    y_cell: np.ndarray
    X_augmented: np.ndarray
    y_augmented: np.ndarray
    teacher_name: str


@dataclass
class CoSupervisionResult:
    kmeans: KMeans
    cell_models: list[CellModel]
    teacher_pool: dict[str, object]
    teacher_scores: dict[str, float]
    classification: bool

    @property
    def centroids(self) -> np.ndarray:
        return np.vstack([cell.centroid for cell in self.cell_models])

    @property
    def selected_teacher_names(self) -> list[str]:
        return [cell.teacher_name for cell in self.cell_models]

    @property
    def augmented_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        X_parts = [cell.X_augmented for cell in self.cell_models]
        y_parts = [cell.y_augmented for cell in self.cell_models]
        return np.vstack(X_parts), np.concatenate(y_parts)

    def predict_initial_mlm(self, X: np.ndarray) -> np.ndarray:
        labels = self.kmeans.predict(X)
        scores = np.zeros(X.shape[0], dtype=float)
        for idx, label in enumerate(labels):
            scores[idx] = self.cell_models[int(label)].model.predict_score(X[idx : idx + 1])[0]
        return scores


def default_teacher(classification: bool):
    if classification:
        return RandomForestClassifier(n_estimators=100, random_state=0)
    return RandomForestRegressor(n_estimators=100, random_state=0)


def normalize_teachers(teachers, classification: bool) -> dict[str, object]:
    if teachers is None:
        return {"RF": default_teacher(classification)}
    if isinstance(teachers, dict):
        return {str(name): clone(model) for name, model in teachers.items()}
    if isinstance(teachers, (list, tuple)):
        return {f"teacher_{idx}": clone(model) for idx, model in enumerate(teachers)}
    return {"teacher": clone(teachers)}


def fit_teacher_pool(teachers, X: np.ndarray, y: np.ndarray, classification: bool):
    teacher_pool = normalize_teachers(teachers, classification)
    fitted = {}
    scores = {}
    for name, teacher in teacher_pool.items():
        teacher.fit(X, y)
        fitted[name] = teacher
        preds = _teacher_scores(teacher, X, classification)
        scores[name] = score_predictions(y, preds, classification)
    return fitted, scores


def _teacher_scores(teacher, X: np.ndarray, classification: bool) -> np.ndarray:
    if classification and hasattr(teacher, "predict_proba"):
        return teacher.predict_proba(X)[:, 1]
    return np.asarray(teacher.predict(X), dtype=float)


def _teacher_quality(y_true: np.ndarray, scores: np.ndarray, classification: bool) -> float:
    if classification:
        labels = (scores >= 0.5).astype(int)
        return float(np.mean(labels == y_true))
    return -float(np.sqrt(np.mean((np.asarray(y_true, dtype=float) - scores) ** 2)))


def _covariance_matrix(X: np.ndarray, covariance_type: str) -> np.ndarray:
    p = X.shape[1]
    if X.shape[0] <= 1:
        return np.eye(p) * 1e-6
    if covariance_type == "diag":
        var = np.var(X, axis=0, ddof=1)
        return np.diag(np.maximum(var, 1e-6))
    cov = np.cov(X.T)
    return cov + np.eye(p) * 1e-6


def augment_cell(
    X_cell: np.ndarray,
    teacher,
    classification: bool,
    num_aug: int,
    covariance_type: str,
    random_state: int,
):
    rng = np.random.RandomState(random_state)
    mean = np.mean(X_cell, axis=0)
    cov = _covariance_matrix(X_cell, covariance_type)
    X_synth = rng.multivariate_normal(mean=mean, cov=cov, size=num_aug)
    if classification and hasattr(teacher, "predict_proba"):
        y_synth = (teacher.predict_proba(X_synth)[:, 1] >= 0.5).astype(int)
    else:
        y_synth = np.asarray(teacher.predict(X_synth))
    return X_synth, y_synth


def select_teacher_for_cell(
    X_cell: np.ndarray,
    y_cell: np.ndarray,
    teacher_pool: dict[str, object],
    whole_scores: dict[str, float],
    classification: bool,
):
    best_name = None
    best_quality = None
    for name, teacher in teacher_pool.items():
        scores = _teacher_scores(teacher, X_cell, classification)
        quality = _teacher_quality(y_cell, scores, classification)
        if best_name is None or quality > best_quality + 1e-12:
            best_name = name
            best_quality = quality
        elif abs(quality - best_quality) <= 1e-12 and whole_scores[name] > whole_scores[best_name]:
            best_name = name
            best_quality = quality
    return best_name


def build_co_supervision(
    X: np.ndarray,
    y: np.ndarray,
    teachers,
    n_cells: int,
    classification: bool,
    num_aug: int,
    covariance_type: str,
    local_model_alpha: float,
    local_model_max_iter: int,
    random_state: int,
) -> CoSupervisionResult:
    teacher_pool, whole_scores = fit_teacher_pool(teachers, X, y, classification)
    kmeans = KMeans(n_clusters=n_cells, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(X)
    cell_models = []

    for cell_id in range(n_cells):
        idx = np.where(labels == cell_id)[0]
        X_cell = X[idx]
        y_cell = y[idx]
        teacher_name = select_teacher_for_cell(
            X_cell,
            y_cell,
            teacher_pool,
            whole_scores,
            classification,
        )
        teacher = teacher_pool[teacher_name]
        X_synth, y_synth = augment_cell(
            X_cell,
            teacher,
            classification,
            num_aug=num_aug,
            covariance_type=covariance_type,
            random_state=random_state + cell_id,
        )
        X_aug = np.vstack([X_cell, X_synth])
        y_aug = np.concatenate([y_cell, y_synth])
        model = make_statsmodels_model(
            classification=classification,
            alpha=local_model_alpha,
            max_iter=local_model_max_iter,
        )
        model.fit(X_aug, y_aug)
        cell_models.append(
            CellModel(
                centroid=kmeans.cluster_centers_[cell_id],
                model=model,
                X_cell=X_cell,
                y_cell=y_cell,
                X_augmented=X_aug,
                y_augmented=y_aug,
                teacher_name=teacher_name,
            )
        )

    return CoSupervisionResult(
        kmeans=kmeans,
        cell_models=cell_models,
        teacher_pool=teacher_pool,
        teacher_scores=whole_scores,
        classification=classification,
    )


def select_num_cells(
    X: np.ndarray,
    y: np.ndarray,
    teachers,
    n_cells_grid,
    classification: bool,
    selection: str,
    num_aug: int,
    covariance_type: str,
    local_model_alpha: float,
    local_model_max_iter: int,
    random_state: int,
):
    if selection == "silhouette":
        best_j = None
        best_score = None
        for n_cells in n_cells_grid:
            if n_cells <= 1 or n_cells >= len(X):
                continue
            kmeans = KMeans(n_clusters=n_cells, random_state=random_state, n_init="auto")
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            if best_j is None or score > best_score:
                best_j = n_cells
                best_score = score
        return best_j, {"selection": "silhouette", "score": best_score}

    X_sub, X_val, y_sub, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y if classification else None,
    )
    best_j = None
    best_objective = None
    results = []
    for n_cells in n_cells_grid:
        if n_cells <= 1 or n_cells >= len(X_sub):
            continue
        cosup = build_co_supervision(
            X_sub,
            y_sub,
            teachers=teachers,
            n_cells=n_cells,
            classification=classification,
            num_aug=num_aug,
            covariance_type=covariance_type,
            local_model_alpha=local_model_alpha,
            local_model_max_iter=local_model_max_iter,
            random_state=random_state,
        )
        scores = cosup.predict_initial_mlm(X_val)
        objective = validation_objective(y_val, scores, classification)
        metric = score_predictions(y_val, scores, classification)
        results.append({"n_cells": n_cells, "metric": metric, "objective": objective})
        if best_j is None or objective > best_objective:
            best_j = n_cells
            best_objective = objective
    return best_j, {"selection": "validation_mlm", "results": results}
