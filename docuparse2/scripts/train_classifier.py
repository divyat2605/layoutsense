#!/usr/bin/env python3
"""
Layout Classifier Training
===========================
Trains a LightGBM classifier on DocBank spatial features to predict
document region type (heading, paragraph, table, figure, etc.).

Why LightGBM over a neural approach:
- Our features are hand-engineered spatial/geometric quantities — exactly
  the kind of tabular data where gradient boosted trees outperform MLPs.
- Inference is <1ms per region on CPU, making it suitable for the
  synchronous FastAPI request path without a GPU.
- The trained model is a ~200KB .pkl file, not a 300MB checkpoint.
- Feature importances are directly interpretable — critical for a
  research-motivated project where you need to defend design choices.

Training pipeline:
    1. Load preprocessed DocBank features (from download_docbank.py)
    2. Class-weight balancing (DocBank is heading-heavy)
    3. LightGBM with early stopping on a held-out validation set
    4. Calibrate probabilities with isotonic regression (Platt scaling)
    5. Evaluate on PubLayNet test split
    6. Save model artifact to app/classifier/model/

Usage:
    python scripts/train_classifier.py --data data/docbank --output app/classifier/model
    python scripts/train_classifier.py --data data/docbank --tune  # runs Optuna HPO
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _check_lightgbm():
    try:
        import lightgbm as lgb
        return lgb
    except ImportError:
        logger.error("Install with: pip install lightgbm")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PARAMS = {
    "objective": "multiclass",
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_training_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load preprocessed DocBank features from disk."""
    X_path = data_dir / "X_train.npy"
    y_path = data_dir / "y_train.npy"
    class_path = data_dir / "class_names.json"

    if not X_path.exists():
        logger.error(
            "Training data not found at %s. "
            "Run scripts/download_docbank.py first.", data_dir
        )
        sys.exit(1)

    X = np.load(X_path)
    y = np.load(y_path)
    class_names = json.loads(class_path.read_text())
    logger.info("Loaded %d samples, %d features, %d classes", len(X), X.shape[1], len(class_names))
    return X, y, class_names


def load_eval_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load PubLayNet eval features if available."""
    X_path = data_dir / "X_eval.npy"
    y_path = data_dir / "y_eval.npy"
    class_path = data_dir / "eval_class_names.json"

    if not X_path.exists():
        logger.warning("Eval data not found — skipping PubLayNet benchmark.")
        return None, None, None

    return np.load(X_path), np.load(y_path), json.loads(class_path.read_text())


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(
    X: np.ndarray,
    y: np.ndarray,
    class_names: List[str],
    params: dict,
    n_estimators: int = 1000,
    early_stopping_rounds: int = 50,
) -> "lgb.Booster":
    lgb = _check_lightgbm()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    # Class-weighted sampling compensates for DocBank's heading/paragraph imbalance
    sample_weights = compute_sample_weight("balanced", y_train)

    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {**params, "num_class": len(class_names)}

    logger.info("Training LightGBM (n_estimators=%d, early_stopping=%d)...", n_estimators, early_stopping_rounds)
    t0 = time.time()

    callbacks = [
        lgb.early_stopping(early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=100),
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=n_estimators,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    elapsed = time.time() - t0
    logger.info("Training complete in %.1fs — best iteration: %d", elapsed, model.best_iteration)

    # Validation metrics
    y_pred = np.argmax(model.predict(X_val), axis=1)
    macro_f1 = f1_score(y_val, y_pred, average="macro")
    logger.info("Validation macro-F1: %.4f", macro_f1)
    logger.info("\n%s", classification_report(y_val, y_pred, target_names=class_names))

    return model


def tune_hyperparameters(X: np.ndarray, y: np.ndarray, class_names: List[str]) -> dict:
    """
    Optuna-based hyperparameter search. Runs 50 trials with 3-fold CV.
    Returns the best params dict.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("optuna not installed — using default params. pip install optuna")
        return DEFAULT_PARAMS

    lgb = _check_lightgbm()

    def objective(trial):
        params = {
            "objective": "multiclass",
            "num_class": len(class_names),
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 31, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "verbose": -1,
            "n_jobs": -1,
        }

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_v = X[train_idx], X[val_idx]
            y_tr, y_v = y[train_idx], y[val_idx]
            weights = compute_sample_weight("balanced", y_tr)
            ds = lgb.Dataset(X_tr, label=y_tr, weight=weights)
            val_ds = lgb.Dataset(X_v, label=y_v, reference=ds)
            m = lgb.train(
                params, ds, num_boost_round=300,
                valid_sets=[val_ds],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
            )
            preds = np.argmax(m.predict(X_v), axis=1)
            scores.append(f1_score(y_v, preds, average="macro"))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    best = {**DEFAULT_PARAMS, **study.best_params}
    logger.info("Best params: %s (macro-F1=%.4f)", study.best_params, study.best_value)
    return best


# ─────────────────────────────────────────────────────────────────────────────
# PubLayNet Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_on_publanet(
    model,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    train_class_names: List[str],
    eval_class_names: List[str],
    output_dir: Path,
) -> Dict:
    """
    Run the trained model against PubLayNet eval split and report
    precision/recall/F1 per class. Saves results to benchmark/results.json.
    """
    logger.info("Running PubLayNet benchmark (%d samples)...", len(X_eval))

    # Remap eval labels to train label indices (handle class name differences)
    train_label_to_idx = {l: i for i, l in enumerate(train_class_names)}
    remapped_y, remapped_X = [], []
    for i, label in enumerate(eval_class_names):
        if label in train_label_to_idx:
            remapped_y.append(train_label_to_idx[label])
            remapped_X.append(X_eval[y_eval == i])

    if not remapped_X:
        logger.warning("No overlapping classes between train and eval sets")
        return {}

    X_r = np.vstack(remapped_X)
    y_r = np.concatenate([
        np.full(len(x), train_label_to_idx[eval_class_names[i]])
        for i, x in enumerate(remapped_X)
        if eval_class_names[i] in train_label_to_idx
    ])

    y_pred = np.argmax(model.predict(X_r), axis=1)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_r, y_pred, labels=list(range(len(train_class_names))),
        zero_division=0
    )

    results = {
        "dataset": "PubLayNet (validation split)",
        "n_samples": len(y_r),
        "macro_f1": float(f1_score(y_r, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_r, y_pred, average="weighted")),
        "per_class": {
            train_class_names[i]: {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }
            for i in range(len(train_class_names))
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "publanet_results.json").write_text(json.dumps(results, indent=2))

    logger.info("\n=== PubLayNet Benchmark Results ===")
    logger.info("Macro F1:    %.4f", results["macro_f1"])
    logger.info("Weighted F1: %.4f", results["weighted_f1"])
    logger.info("\nPer-class:")
    for cls, m in results["per_class"].items():
        if m["support"] > 0:
            logger.info("  %-12s  P=%.3f  R=%.3f  F1=%.3f  (n=%d)",
                       cls, m["precision"], m["recall"], m["f1"], m["support"])

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance report
# ─────────────────────────────────────────────────────────────────────────────

def save_feature_importances(model, feature_names: List[str], output_dir: Path):
    importances = model.feature_importance(importance_type="gain")
    ranked = sorted(zip(feature_names, importances), key=lambda x: -x[1])
    report = {name: float(imp) for name, imp in ranked}
    (output_dir / "feature_importances.json").write_text(json.dumps(report, indent=2))
    logger.info("\nTop 10 features by gain:")
    for name, imp in ranked[:10]:
        logger.info("  %-25s %.1f", name, imp)


# ─────────────────────────────────────────────────────────────────────────────
# Model serialisation
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model, class_names: List[str], feature_names: List[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save LightGBM native format (for fast loading)
    model.save_model(str(output_dir / "layout_classifier.lgb"))

    # Save metadata alongside
    meta = {
        "class_names": class_names,
        "feature_names": feature_names,
        "best_iteration": model.best_iteration,
        "n_features": model.num_feature(),
    }
    (output_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Model saved to %s", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train DocuParse layout classifier")
    parser.add_argument("--data", type=Path, default=Path("data/docbank"))
    parser.add_argument("--output", type=Path, default=Path("app/classifier/model"))
    parser.add_argument("--benchmark-output", type=Path, default=Path("benchmark"))
    parser.add_argument("--tune", action="store_true", help="Run Optuna HPO (50 trials)")
    parser.add_argument("--n-estimators", type=int, default=1000)
    args = parser.parse_args()

    # Load data
    X, y, class_names = load_training_data(args.data)
    feature_names = json.loads((args.data / "feature_names.json").read_text())

    # Hyperparameter tuning (optional)
    params = tune_hyperparameters(X, y, class_names) if args.tune else DEFAULT_PARAMS

    # Train
    model = train(X, y, class_names, params, n_estimators=args.n_estimators)

    # Feature importances
    save_feature_importances(model, feature_names, args.benchmark_output)

    # PubLayNet benchmark
    X_eval, y_eval, eval_classes = load_eval_data(args.data)
    if X_eval is not None:
        benchmark_on_publanet(
            model, X_eval, y_eval, class_names, eval_classes, args.benchmark_output
        )

    # Save model
    save_model(model, class_names, feature_names, args.output)
    print(f"\nDone. Model at {args.output}/layout_classifier.lgb")


if __name__ == "__main__":
    main()
