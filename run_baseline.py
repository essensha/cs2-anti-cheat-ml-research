"""Reproducible synthetic baseline: temporal split, models, fairness and calibration.

No network, external data, production deployment or enforcement.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["skill_proxy", "actions_per_min", "median_reaction_ms",
            "timing_regularities", "target_accuracy", "repeated_action_ratio"]
LABEL = "is_suspicious_simulated"

def run(data_path, output_dir, threshold=0.7):
    df = pd.read_csv(data_path)
    needed = set(FEATURES + [LABEL, "match_day", "player_id", "match_id"])
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df[list(needed)].isna().any().any():
        raise ValueError("Missing values in required fields")
    days = np.sort(df.match_day.unique())
    if len(days) < 5:
        raise ValueError("Need at least five unique days")
    cutoff = days[int(len(days) * 0.8)]
    train, test = df[df.match_day < cutoff].copy(), df[df.match_day >= cutoff].copy()
    if train.empty or test.empty or train[LABEL].nunique() < 2 or test[LABEL].nunique() < 2:
        raise ValueError("Temporal train/test split must have both classes")
    if set(train.match_id) & set(test.match_id):
        raise ValueError("Match identifiers leak across time split")
    if set(train.player_id) & set(test.player_id):
        raise ValueError("Player identifiers overlap across time split; review evaluation design")
    xtrain, xtest, ytrain, ytest = train[FEATURES], test[FEATURES], train[LABEL], test[LABEL]
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    lr.fit(xtrain, ytrain)
    prob = lr.predict_proba(xtest)[:, 1]
    iso = make_pipeline(StandardScaler(), IsolationForest(n_estimators=200, contamination=0.07, random_state=42))
    iso.fit(xtrain)
    anomaly = -iso.decision_function(xtest)
    prediction = (prob >= threshold).astype(int)
    metrics = {
        "dataset": "synthetic_generated", "train_rows": int(len(train)), "test_rows": int(len(test)),
        "cutoff_day": int(cutoff), "test_prevalence": float(ytest.mean()), "threshold": threshold,
        "logistic_regression": {
            "roc_auc": float(roc_auc_score(ytest, prob)),
            "average_precision": float(average_precision_score(ytest, prob)),
            "brier": float(brier_score_loss(ytest, prob)),
            "precision": float(precision_score(ytest, prediction, zero_division=0)),
            "recall": float(recall_score(ytest, prediction, zero_division=0))
        },
        "isolation_forest": {
            "roc_auc": float(roc_auc_score(ytest, anomaly)),
            "average_precision": float(average_precision_score(ytest, anomaly))
        }
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    evaluation = test[["player_id", "match_day", "skill_proxy", LABEL]].copy()
    evaluation["predicted_probability"] = prob
    evaluation["predicted_label"] = prediction
    evaluation["skill_group"] = pd.qcut(evaluation.skill_proxy, q=5, labels=["Very Low", "Low", "Medium", "High", "Very High"])
    groups = []
    for group, subset in evaluation.groupby("skill_group", observed=True):
        normal = subset[subset[LABEL] == 0]
        fp = int(normal.predicted_label.sum())
        groups.append({"skill_group": str(group), "observations": int(len(subset)),
                       "normal_observations": int(len(normal)), "false_positives": fp,
                       "false_positive_rate": float(fp / len(normal)) if len(normal) else None})
    metrics["skill_segments"] = groups
    evaluation.to_csv(out / "predictions.csv", index=False)
    with (out / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    return metrics

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/synthetic_anticheat_telemetry.csv")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--threshold", type=float, default=0.7)
    args = p.parse_args()
    if not 0 <= args.threshold <= 1:
        p.error("threshold must be between 0 and 1")
    run(args.data, args.output_dir, args.threshold)
