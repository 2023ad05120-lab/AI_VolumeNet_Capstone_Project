import os, argparse, numpy as np
from PIL import Image
import json 

def load_depth_map(path):
    return np.array(Image.open(path)).astype(np.float32)

def main(pred_dir, gt_dir, output_path):
    pred_files = sorted(os.listdir(pred_dir))
    gt_files = sorted(os.listdir(gt_dir))

    rmse_list, mae_list, delta_acc = [], [], []

    for pf, gf in zip(pred_files, gt_files):
        pred = load_depth_map(os.path.join(pred_dir, pf))
        gt = load_depth_map(os.path.join(gt_dir, gf))

        if pred.shape != gt.shape:
            continue

        diff = pred - gt
        rmse = np.sqrt(np.mean(diff**2))
        mae = np.mean(np.abs(diff))

        ratio = np.maximum(pred / gt, gt / pred)
        delta1 = np.mean(ratio < 1.25)
        delta2 = np.mean(ratio < 1.25**2)
        delta3 = np.mean(ratio < 1.25**3)

        rmse_list.append(rmse)
        mae_list.append(mae)
        delta_acc.append((delta1, delta2, delta3))

    results = {
        "samples_evaluated": len(rmse_list),
        "rmse_mean": float(np.mean(rmse_list)) if rmse_list else None,
        "mae_mean": float(np.mean(mae_list)) if mae_list else None,
        "delta1": float(np.mean([d[0] for d in delta_acc])) if delta_acc else None,
        "delta2": float(np.mean([d[1] for d in delta_acc])) if delta_acc else None,
        "delta3": float(np.mean([d[2] for d in delta_acc])) if delta_acc else None
    }

    print(json.dumps(results, indent=2))
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, required=True)
    parser.add_argument("--gt", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    main(args.pred, args.gt, args.output)
