import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime

def ensure_paths():
    eval_dir = os.path.abspath(os.path.join("outputs", "evaluation"))
    pred_dir = os.path.abspath(os.path.join("outputs", "predictions"))
    os.makedirs(eval_dir, exist_ok=True)
    return eval_dir, pred_dir

def load_per_sample_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}. Run evaluate_depth.py first.")
    df = pd.read_csv(csv_path)
    # Normalize column names just in case
    df.columns = [c.strip() for c in df.columns]
    # Extract gesture from ground_truth_file path
    df["gesture"] = df["ground_truth_file"].apply(
        lambda x: x.split("/")[0] if "/" in x else x.split("\\")[0]
    )
    return df

def scan_predictions(pred_dir):
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.png")))
    return [os.path.basename(p) for p in pred_files]

def coverage_analysis(df, pred_basenames):
    evaluated = set(df["prediction_file"].tolist())
    all_preds = set(pred_basenames)
    missing = sorted(list(all_preds - evaluated))
    coverage = len(evaluated) / max(1, len(all_preds))
    return coverage, missing

def gesture_stats(df):
    numeric_cols = ["rmse", "mae", "delta1", "delta2", "delta3"]
    stats_mean = df.groupby("gesture")[numeric_cols].mean().round(4)
    stats_std = df.groupby("gesture")[numeric_cols].std(ddof=1).round(4)
    return stats_mean, stats_std

def top_bottom(df, metric="rmse", k=10):
    df_sorted = df.sort_values(by=metric, ascending=True)
    best = df_sorted.head(k)
    worst = df_sorted.tail(k).sort_values(by=metric, ascending=False)
    return best, worst

def detect_outliers(df, metric="rmse", z_thresh=2.0):
    mu = df[metric].mean()
    sigma = df[metric].std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        return pd.DataFrame(columns=df.columns)
    df = df.copy()
    df["zscore"] = (df[metric] - mu) / sigma
    outliers = df[df["zscore"].abs() > z_thresh].sort_values(by="zscore", ascending=False)
    return outliers

def df_to_md_table(df, cols):
    # Limit rows to avoid overly large tables in Markdown
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        vals = []
        for c in cols:
            val = row[c]
            if isinstance(val, float):
                val = f"{val:.4f}"
            vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)

def save_gesture_csv(eval_dir, stats_mean):
    out_csv = os.path.join(eval_dir, "gesture_wise_metrics.csv")
    stats_mean.to_csv(out_csv)

def generate_report(eval_dir, df, pred_basenames):
    report_path = os.path.join(eval_dir, "report.md")
    coverage, missing = coverage_analysis(df, pred_basenames)
    stats_mean, stats_std = gesture_stats(df)
    best_rmse, worst_rmse = top_bottom(df, "rmse", 10)
    best_mae, worst_mae = top_bottom(df, "mae", 10)
    outliers_rmse = detect_outliers(df, "rmse", 2.0)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build Markdown
    md = []
    md.append(f"# AI VolumeNet evaluation report")
    md.append("")
    md.append(f"**Generated:** {timestamp}")
    md.append(f"**Source:** outputs/evaluation/per_sample_metrics.csv")
    md.append("---")

    # Coverage
    md.append("## Coverage analysis")
    md.append(f"- **Total predictions:** {len(pred_basenames)}")
    md.append(f"- **Evaluated predictions:** {df.shape[0]}")
    md.append(f"- **Coverage:** {coverage:.2%}")
    if missing:
        md.append("")
        md.append("### Missing predictions (no ground truth match)")
        md.append(df_to_md_table(pd.DataFrame({"prediction_file": missing}), ["prediction_file"]))
    md.append("---")

    # Gesture stats
    md.append("## Gesture-wise metrics (mean)")
    stats_mean_reset = stats_mean.reset_index()
    md.append(df_to_md_table(stats_mean_reset, ["gesture", "rmse", "mae", "delta1", "delta2", "delta3"]))
    md.append("")
    md.append("## Gesture-wise variability (std dev)")
    stats_std_reset = stats_std.reset_index()
    md.append(df_to_md_table(stats_std_reset, ["gesture", "rmse", "mae", "delta1", "delta2", "delta3"]))
    md.append("---")

    # Top/Bottom by RMSE
    md.append("## Top 10 files by lowest RMSE (best)")
    md.append(df_to_md_table(best_rmse.reset_index(drop=True),
                             ["prediction_file", "ground_truth_file", "gesture", "rmse", "mae", "delta1", "delta2", "delta3"]))
    md.append("")
    md.append("## Top 10 files by highest RMSE (worst)")
    md.append(df_to_md_table(worst_rmse.reset_index(drop=True),
                             ["prediction_file", "ground_truth_file", "gesture", "rmse", "mae", "delta1", "delta2", "delta3"]))
    md.append("---")

    # Top/Bottom by MAE
    md.append("## Top 10 files by lowest MAE (best)")
    md.append(df_to_md_table(best_mae.reset_index(drop=True),
                             ["prediction_file", "ground_truth_file", "gesture", "rmse", "mae", "delta1", "delta2", "delta3"]))
    md.append("")
    md.append("## Top 10 files by highest MAE (worst)")
    md.append(df_to_md_table(worst_mae.reset_index(drop=True),
                             ["prediction_file", "ground_truth_file", "gesture", "rmse", "mae", "delta1", "delta2", "delta3"]))
    md.append("---")

    # Outliers
    md.append("## Outliers by RMSE (|z| > 2.0)")
    if outliers_rmse.empty:
        md.append("> No RMSE outliers detected at z-threshold 2.0.")
    else:
        md.append(df_to_md_table(outliers_rmse.reset_index(drop=True),
                                 ["prediction_file", "ground_truth_file", "gesture", "rmse", "mae", "delta1", "delta2", "delta3", "zscore"]))
    md.append("---")

    # Full per-file table (capped to avoid huge rendering; remove head limit if desired)
    md.append("## Per-file metrics (full)")
    md.append(df_to_md_table(df.reset_index(drop=True),
                             ["prediction_file", "ground_truth_file", "gesture", "rmse", "mae", "delta1", "delta2", "delta3"]))
    md.append("---")

    # Recommendations based on findings
    md.append("## Recommendations")
    md.append("- **Calibration:** Prioritize scale refinement for gestures/classes appearing in worst-10 lists; inspect `outputs/evaluation/vis/` diffs.")
    md.append("- **Augmentation:** Upsample weak gestures and challenging poses; add motion blur/occlusion augmentations for scroll-type gestures.")
    md.append("- **Modeling:** Consider per-class calibration factors or post-hoc regression fine-tuning on YCB for volume estimation.")
    md.append("- **Pipeline:** Batch inference and ONNX/TensorRT conversion for faster turnaround on evaluation cycles.")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    return report_path, stats_mean

def main():
    eval_dir, pred_dir = ensure_paths()
    csv_path = os.path.join(eval_dir, "per_sample_metrics.csv")
    df = load_per_sample_csv(csv_path)
    pred_basenames = scan_predictions(pred_dir)
    report_path, stats_mean = generate_report(eval_dir, df, pred_basenames)
    # Save gesture-wise means to CSV for downstream references
    save_gesture_csv(eval_dir, stats_mean)
    print(f"✅ Report generated: {report_path}")
    print(f"✅ Gesture-wise metrics saved: {os.path.join(eval_dir, 'gesture_wise_metrics.csv')}")

if __name__ == "__main__":
    main()
