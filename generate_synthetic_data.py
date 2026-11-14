"""Generate reproducible, fully synthetic CS2-inspired player telemetry.

Not Valve data. Labels encode planted patterns, not confirmed cheating.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

COLUMNS = ["player_id", "match_id", "match_day", "skill_proxy", "actions_per_min",
           "median_reaction_ms", "timing_regularities", "target_accuracy",
           "repeated_action_ratio", "synthetic_pattern", "is_suspicious_simulated"]

def generate(n=12000, seed=42, days=60):
    if n < 100 or days < 10:
        raise ValueError("n must be >= 100 and days >= 10")
    rng = np.random.default_rng(seed)
    day = rng.integers(0, days, n)
    skill = rng.uniform(0, 1, n)
    suspicious = rng.binomial(1, 0.065, n)
    style = np.where(suspicious == 0, "none",
                     np.where(rng.random(n) < 0.5, "fast_reaction", "precision_pattern"))
    reaction = 290 - 72 * skill - 65 * (style == "fast_reaction") - 18 * (style == "precision_pattern") + rng.normal(0, 22, n)
    accuracy = np.clip(0.27 + 0.36 * skill + 0.23 * (style == "precision_pattern") + 0.12 * (style == "fast_reaction") + rng.normal(0, 0.07, n), 0, 1)
    regularity = np.clip(0.23 + 0.17 * skill + 0.38 * suspicious + rng.normal(0, 0.11, n), 0, 1)
    repeated = np.clip(0.13 + 0.09 * skill + 0.31 * suspicious + rng.normal(0, 0.09, n), 0, 1)
    apm = np.maximum(1, 90 + 90 * skill + 40 * suspicious + rng.normal(0, 25, n))
    return pd.DataFrame({
        "player_id": np.arange(100000, 100000 + n),
        "match_id": np.arange(200000, 200000 + n),
        "match_day": day,
        "skill_proxy": skill.round(5),
        "actions_per_min": apm.round(3),
        "median_reaction_ms": reaction.clip(50, 600).round(3),
        "timing_regularities": regularity.round(5),
        "target_accuracy": accuracy.round(5),
        "repeated_action_ratio": repeated.round(5),
        "synthetic_pattern": style,
        "is_suspicious_simulated": suspicious
    })[COLUMNS]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=12000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="data/synthetic_anticheat_telemetry.csv")
    a = p.parse_args()
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate(a.rows, a.seed)
    df.to_csv(out, index=False)
    print(f"Generated {len(df)} synthetic rows to {out}; suspicious rate={df.is_suspicious_simulated.mean():.3f}")
