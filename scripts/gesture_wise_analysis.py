import pandas as pd
import os

def main():
    # ✅ Path to the per-sample metrics CSV
    csv_path = os.path.abspath("outputs/evaluation/per_sample_metrics.csv")

    if not os.path.exists(csv_path):
        print(f"⚠️ CSV not found at {csv_path}. Run evaluate_depth.py first.")
        return

    # Load CSV
    df = pd.read_csv(csv_path)

    # ✅ Extract gesture name from ground_truth_file path
    # Example: "scroll_left/2021-10-16-124936_0/1.png" → "scroll_left"
    df["gesture"] = df["ground_truth_file"].apply(
        lambda x: x.split("/")[0] if "/" in x else x.split("\\")[0]
    )

    # Numeric columns to aggregate
    numeric_cols = ["rmse", "mae", "delta1", "delta2", "delta3"]

    # ✅ Group by gesture and compute averages
    gesture_stats = df.groupby("gesture")[numeric_cols].mean()

    # Print results
    print("----- Gesture-wise Metrics -----")
    print(gesture_stats)

    # ✅ Save gesture-wise metrics to CSV
    out_path = os.path.abspath("outputs/evaluation/gesture_wise_metrics.csv")
    gesture_stats.to_csv(out_path)
    print(f"\n✅ Gesture-wise metrics saved to {out_path}")

if __name__ == "__main__":
    main()
