# metrics_model.py
# -*- coding: utf-8 -*-
"""
Обучение бустинга по табличным метрикам сигналов и расширенная аналитика.

API (минимум):
--------------
from metrics_model import train_metrics_model, predict_label, predict_proba

model = train_metrics_model(records)  # records — ваш список словарей (см. пример в условии)
p = predict_proba(model, records[0]["metrics"])
y = predict_label(model, records[0]["metrics"])

Что возвращает train_metrics_model:
-----------------------------------
Dataclass MetricsModelBundle с полями:
- pipeline: обученный sklearn Pipeline (imputer + GradientBoostingClassifier)
- feature_names: список имён признаков (порядок обучения)
- threshold: используемый порог (по умолчанию 0.5; опционально можно выставить по Youden/F1)
- classes_: np.ndarray([0, 1])
- cv_table: DataFrame по кросс-валидации (fold-by-fold метрики)
- test_report: dict с метриками на holdout-тесте
- confusion_matrix: np.ndarray(2x2) для holdout-теста
- feature_importance: DataFrame (встроенная важность GBDT)
- perm_importance: DataFrame (пермутационная важность на тесте)
- pair_synergy: DataFrame (оценка полезности пар признаков)
- meta: словарь с технической информацией

Заметки:
--------
- В качестве целевой метки используется 1, если profit > 0, иначе 0. Значения profit == 0
  исключаются из обучения (редкость и не несут информации об знаке).
- Для устойчивости: константные признаки отбрасываются (нулевая дисперсия).
- Сильный дисбаланс классов компенсируется sample_weight при финальном обучении.
- Кросс-валидация: StratifiedKFold, по умолчанию 5 фолдов; дополнительно делается holdout-сплит.
- Парные взаимодействия: для каждой пары признаков обучается дерево глубины 2 и сравнивается
  ROC AUC пары с лучшим ROC AUC одиночных деревьев глубины 1 по каждому из признаков. Разность
  и есть «synergy» (чем выше, тем сильнее взаимодействие).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier


# =========================
# ВСПОМОГАТЕЛЬНЫЕ ПРОЦЕДУРЫ
# =========================

def _records_to_dataframe(
    records: List[Dict[str, Any]],
    drop_profit_eq_zero: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Преобразует список словарей в X (DataFrame признаков), y (Series target), meta (open/close_time).
    - Извлекает X из вложенных словарей records[i]['metrics'].
    - y = 1, если profit > 0; y = 0, если profit < 0. profit == 0 — опционально выкидывается.
    - Возвращает также meta с полями open_time/close_time/type_of_signal/type_of_close/profit.
    """
    if not isinstance(records, list) or not records:
        raise ValueError("records должен быть непустым списком словарей.")

    # Собираем таблицы
    metrics_list = []
    meta_list = []
    y_list = []

    for rec in records:
        if "metrics" not in rec or not isinstance(rec["metrics"], dict):
            raise ValueError("Каждый элемент должен содержать словарь 'metrics'.")
        metrics_list.append(rec["metrics"])

        prof = rec.get("profit", None)
        if prof is None:
            raise ValueError("В каждом элементе должен быть 'profit' (float).")

        if prof == 0 and drop_profit_eq_zero:
            # Маркируем NaN, выбросим позже
            y_list.append(np.nan)
        else:
            y_list.append(1 if prof > 0 else 0)

        meta_list.append({
            "open_time": rec.get("open_time"),
            "close_time": rec.get("close_time"),
            "type_of_signal": rec.get("type_of_signal"),
            "type_of_close": rec.get("type_of_close"),
            "profit": prof,
        })

    X = pd.DataFrame(metrics_list)
    y = pd.Series(y_list, name="target").astype("float")

    # Удалим строки с y==NaN (profit == 0, если так настроено)
    mask = ~y.isna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].astype(int).reset_index(drop=True)
    meta = pd.DataFrame(meta_list).loc[mask].reset_index(drop=True)

    # Приведём типы признаков к числовым (на всякий случай) и отсортируем колонки
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.reindex(sorted(X.columns), axis=1)

    # Выкинем константные признаки
    nunq = X.nunique(dropna=True)
    const_cols = nunq[nunq <= 1].index.tolist()
    if const_cols:
        X = X.drop(columns=const_cols)

    # Если после фильтрации нет столбцов — ошибка
    if X.shape[1] == 0:
        raise ValueError("После удаления константных признаков не осталось ни одного признака.")

    return X, y, meta


def _compute_class_weights(y: pd.Series) -> np.ndarray:
    """
    Возвращает вектор sample_weight для балансировки классов: вес ~ 1 / freq(class).
    """
    values, counts = np.unique(y, return_counts=True)
    freq = {v: c for v, c in zip(values, counts)}
    total = len(y)
    weights = {cls: total / (len(values) * freq[cls]) for cls in values}
    return np.array([weights[int(t)] for t in y], dtype=float)


def _build_pipeline(random_state: int = 42) -> Pipeline:
    """
    Простая, устойчивая к пропускам линия:
    - Imputer (median)
    - GradientBoostingClassifier (скромные настройки, чтобы не переобучиться при N=100-200)
    """
    clf = GradientBoostingClassifier(
        learning_rate=0.05,
        n_estimators=300,
        max_depth=3,
        subsample=0.8,
        random_state=random_state,
    )
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", clf),
    ])
    return pipe


def _manual_cv(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    random_state: int
) -> pd.DataFrame:
    """
    Ручная стратифицированная КВ с подсчётом Accuracy/Precision/Recall/F1/ROC-AUC по фолдам.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []
    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        # Обучим без sample_weight внутри КВ, чтобы не усложнять (Опционально можно добавить)
        model = _build_pipeline(random_state + fold)
        model.fit(X_tr, y_tr)

        prob = model.predict_proba(X_va)[:, 1]
        pred = (prob >= 0.5).astype(int)

        # Метрики
        acc = accuracy_score(y_va, pred)
        pre = precision_score(y_va, pred, zero_division=0)
        rec = recall_score(y_va, pred, zero_division=0)
        f1 = f1_score(y_va, pred, zero_division=0)
        try:
            auc = roc_auc_score(y_va, prob)
        except Exception:
            auc = np.nan

        rows.append({
            "fold": fold,
            "n_val": int(y_va.shape[0]),
            "acc": acc,
            "precision": pre,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc,
        })

    cv_tbl = pd.DataFrame(rows)
    return cv_tbl


def _feature_importances(
    trained_pipeline: Pipeline,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Возвращает встроенную важность признаков из GBDT.
    """
    clf: GradientBoostingClassifier = trained_pipeline.named_steps["clf"]  # type: ignore
    imp = getattr(clf, "feature_importances_", None)
    if imp is None:
        raise RuntimeError("У классификатора нет feature_importances_.")
    return (
        pd.DataFrame({"feature": feature_names, "importance": imp})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _permutation_importances(
    trained_pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
    n_repeats: int = 50
) -> pd.DataFrame:
    """
    Пермутационная важность на тестовой выборке (стабильнее и интерпретируемее).
    """
    r = permutation_importance(
        trained_pipeline,
        X_test, y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="roc_auc",
        n_jobs=None
    )
    tbl = pd.DataFrame({
        "feature": X_test.columns,
        "perm_importance_mean": r.importances_mean,
        "perm_importance_std": r.importances_std,
    }).sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)
    return tbl


def _pairwise_synergy(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    n_splits: int = 5,
    min_samples_leaf: int = 5
) -> pd.DataFrame:
    """
    Оценивает взаимодействия пар признаков.
    Метод: ROC AUC пары (дерево depth=2) минус лучший ROC AUC одиночных (stump depth=1).
    Положительная «synergy» => пара действительно даёт что-то сверх любого одиночного признака.
    """
    features = list(X.columns)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # AUC для одиночных признаков (деревья-пни)
    single_auc: Dict[str, float] = {}
    for f in features:
        scores = []
        for (tr, va) in skf.split(X, y):
            dt = DecisionTreeClassifier(
                max_depth=1,
                random_state=random_state,
                min_samples_leaf=min_samples_leaf
            )
            dt.fit(X.iloc[tr][[f]], y.iloc[tr])
            prob = dt.predict_proba(X.iloc[va][[f]])[:, 1]
            try:
                scores.append(roc_auc_score(y.iloc[va], prob))
            except Exception:
                scores.append(np.nan)
        single_auc[f] = float(np.nanmean(scores))

    # Пары
    rows = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1, f2 = features[i], features[j]
            scores = []
            for (tr, va) in skf.split(X, y):
                dt2 = DecisionTreeClassifier(
                    max_depth=2,
                    random_state=random_state,
                    min_samples_leaf=min_samples_leaf
                )
                dt2.fit(X.iloc[tr][[f1, f2]], y.iloc[tr])
                prob = dt2.predict_proba(X.iloc[va][[f1, f2]])[:, 1]
                try:
                    scores.append(roc_auc_score(y.iloc[va], prob))
                except Exception:
                    scores.append(np.nan)
            pair_auc = float(np.nanmean(scores))
            best_single = float(max(single_auc[f1], single_auc[f2]))
            rows.append({
                "feat_a": f1,
                "feat_b": f2,
                "auc_pair_mean": pair_auc,
                "auc_best_single": best_single,
                "synergy": pair_auc - best_single
            })

    synergy_tbl = pd.DataFrame(rows).sort_values(
        ["synergy", "auc_pair_mean"], ascending=[False, False]
    ).reset_index(drop=True)
    return synergy_tbl


# ==================
# ОСНОВНЫЕ СТРУКТУРЫ
# ==================

@dataclass
class MetricsModelBundle:
    pipeline: Pipeline
    feature_names: List[str]
    threshold: float
    classes_: np.ndarray

    cv_table: pd.DataFrame
    test_report: Dict[str, Any]
    confusion_matrix: np.ndarray

    feature_importance: pd.DataFrame
    perm_importance: pd.DataFrame
    pair_synergy: pd.DataFrame

    meta: Dict[str, Any]


# =================
# ОСНОВНАЯ ФУНКЦИЯ
# =================

def train_metrics_model(
    records: List[Dict[str, Any]],
    *,
    test_size: float = 0.25,
    random_state: int = 42,
    cv_splits: int = 5,
    perm_repeats: int = 50,
    choose_threshold_by: Optional[str] = None  # None | "youden" | "f1"
) -> MetricsModelBundle:
    """
    Главная точка входа.

    Параметры:
    ----------
    records: список словарей с ключами 'profit' (float) и 'metrics' (dict с признаками).
    test_size: доля тестовой выборки (holdout).
    random_state: сид для воспроизводимости.
    cv_splits: число фолдов стратифицированной КВ.
    perm_repeats: число повторов для пермутационной важности.
    choose_threshold_by: если задано, подберёт порог по обучению.
                         - "youden": по максимуму TPR - FPR (ROC)
                         - "f1": по максимуму F1
                         Если None — используется 0.5.

    Возврат:
    --------
    MetricsModelBundle — обученная модель + расширенная аналитика.
    """
    # 1) Подготовка данных
    X, y, meta = _records_to_dataframe(records, drop_profit_eq_zero=True)

    # 2) Кросс-валидация (оценка стабильности на всей выборке)
    cv_table = _manual_cv(_build_pipeline(random_state), X, y, n_splits=cv_splits, random_state=random_state)

    # 3) Holdout-сплит
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=max(min(test_size, 0.49), 0.1),
        random_state=random_state, stratify=y
    )

    # 4) Обучение финальной модели (с балансировкой весами классов)
    weights_train = _compute_class_weights(y_train)
    pipeline = _build_pipeline(random_state)
    pipeline.fit(X_train, y_train, clf__sample_weight=weights_train)  # type: ignore

    # 5) Подбор порога (опционально)
    threshold = 0.5
    prob_train = pipeline.predict_proba(X_train)[:, 1]
    if choose_threshold_by in {"youden", "f1"}:
        from sklearn.metrics import roc_curve
        if choose_threshold_by == "youden":
            fpr, tpr, thr = roc_curve(y_train, prob_train)
            idx = int(np.argmax(tpr - fpr))
            threshold = float(thr[idx])
        else:  # "f1"
            candidates = np.linspace(0.05, 0.95, 19)
            f1_vals = [f1_score(y_train, (prob_train >= t).astype(int), zero_division=0) for t in candidates]
            threshold = float(candidates[int(np.argmax(f1_vals))])

    # 6) Оценка на тесте
    prob_test = pipeline.predict_proba(X_test)[:, 1]
    pred_test = (prob_test >= threshold).astype(int)

    test_metrics = {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "threshold": threshold,
        "acc": accuracy_score(y_test, pred_test),
        "precision": precision_score(y_test, pred_test, zero_division=0),
        "recall": recall_score(y_test, pred_test, zero_division=0),
        "f1": f1_score(y_test, pred_test, zero_division=0),
        "roc_auc": (roc_auc_score(y_test, prob_test) if len(np.unique(y_test)) > 1 else np.nan),
        "report": classification_report(y_test, pred_test, output_dict=True, zero_division=0),
    }
    cm = confusion_matrix(y_test, pred_test, labels=[0, 1])

    # 7) Важности признаков
    feat_imp = _feature_importances(pipeline, list(X.columns))
    perm_imp = _permutation_importances(pipeline, X_test, y_test, random_state=random_state, n_repeats=perm_repeats)

    # 8) Парные взаимодействия
    synergy_tbl = _pairwise_synergy(X, y, random_state=random_state, n_splits=cv_splits)

    # 9) Упаковка результата
    bundle = MetricsModelBundle(
        pipeline=pipeline,
        feature_names=list(X.columns),
        threshold=threshold,
        classes_=np.array([0, 1], dtype=int),

        cv_table=cv_table,
        test_report=test_metrics,
        confusion_matrix=cm,

        feature_importance=feat_imp,
        perm_importance=perm_imp,
        pair_synergy=synergy_tbl,

        meta={
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "dropped_zero_profit": True,
            "const_features_removed": True,
            "cv_splits": cv_splits,
            "model": "sklearn.GradientBoostingClassifier",
            "random_state": random_state,
        }
    )
    return bundle


# ==========================
# ПРЕДСКАЗАНИЯ ДЛЯ НОВОГО КЕЙСА
# ==========================

def _ensure_feature_order(bundle: MetricsModelBundle, metrics: Dict[str, Any]) -> np.ndarray:
    """
    Преобразует входной словарь признаков в numpy-вектор в порядке bundle.feature_names.
    Отсутствующие признаки -> NaN (импутируются пайплайном).
    Лишние ключи игнорируются.
    """
    row = [pd.to_numeric(metrics.get(f, np.nan), errors="coerce") for f in bundle.feature_names]
    return np.array(row, dtype=float).reshape(1, -1)


def predict_proba(bundle: MetricsModelBundle, metrics: Dict[str, Any]) -> float:
    """
    Возвращает вероятность класса 1 (profit > 0) для одного нового примера (dict metrics).
    """
    X_row = _ensure_feature_order(bundle, metrics)
    prob = float(bundle.pipeline.predict_proba(X_row)[:, 1][0])
    return prob


def predict_label(bundle: MetricsModelBundle, metrics: Dict[str, Any], threshold: Optional[float] = None) -> int:
    """
    Возвращает метку 0/1 для одного нового примера по заданному порогу (по умолчанию bundle.threshold).
    """
    thr = bundle.threshold if threshold is None else float(threshold)
    p = predict_proba(bundle, metrics)
    return int(p >= thr)


# ==================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ==================

# if __name__ == "__main__":
#     # Пример: минимальная проверка на игрушечных данных (замените на реальные records)
#     demo_records = [
#         {
#             "open_time": 1755860400000,
#             "close_time": 1755863999999,
#             "type_of_signal": 1,
#             "type_of_close": "time_close",
#             "profit": 0.01,
#             "duration": 60,
#             "metrics": {"a": 0.1, "b": -1.2, "c": 3.4}
#         },
#         {
#             "open_time": 1755860400000,
#             "close_time": 1755863999999,
#             "type_of_signal": 1,
#             "type_of_close": "time_close",
#             "profit": -0.02,
#             "duration": 60,
#             "metrics": {"a": -0.2, "b": 0.9, "c": 1.5}
#         },
#         # добавьте ещё...
#     ]
#     model = train_metrics_model(demo_records, test_size=0.3, choose_threshold_by="f1")
#     print("Holdout test metrics:", {k: v for k, v in model.test_report.items() if k not in ("report",)})
#     print("Confusion matrix [[TN, FP],[FN, TP]]:\n", model.confusion_matrix)
#     print("Top feature importances:\n", model.feature_importance.head(10))
#     print("Top permutation importances:\n", model.perm_importance.head(10))
#     print("Top pair synergies:\n", model.pair_synergy.head(10))

#     example_metrics = demo_records[0]["metrics"]
#     print("Example proba:", predict_proba(model, example_metrics))
#     print("Example label:", predict_label(model, example_metrics))
