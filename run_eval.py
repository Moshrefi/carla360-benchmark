"""Inspect inputs or run inference. Numerical GT pose scoring is a separate step."""
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
from datasets.carla360 import Carla360Dataset

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", required=True)
    p.add_argument("--pose-csv", required=True)
    p.add_argument("--output", default="outputs/smoke")
    p.add_argument("--limit", type=int, default=3)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--inspect-only", action="store_true")
    p.add_argument("--checkpoint")
    p.add_argument("--device", choices=["cpu","cuda"], default="cuda")
    a = p.parse_args()
    if a.limit < 1 or a.stride < 1:
        p.error("limit and stride must be positive")
    ds = Carla360Dataset(a.image_dir, a.pose_csv)
    records = ds.manifest()[::a.stride][:a.limit]
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        p.error("Output directory must be empty; choose a new run name")
    out.mkdir(parents=True, exist_ok=True)
    meta = {"frames":records,"arguments":vars(a),"status":"inputs_ready",
            "gt_columns":["x","y","z","pitch","yaw","roll"],
            "pose_metrics":"not computed; requires explicit model/ERP coordinate convention"}
    try:
        meta["upstream_commit"] = subprocess.check_output(["git","-C",str(Path(__file__).resolve().parent/"third_party/PanoVGGT"),"rev-parse","HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        meta["upstream_commit"] = None
    def save():
        (out/"manifest.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    save()
    print("Selected frame IDs:",[v["frame_id"] for v in records],flush=True)
    if a.inspect_only:
        for i in range(0,len(ds),a.stride):
            if i//a.stride >= a.limit: break
            item=ds[i]
            print(item["frame_id"],item["rgb"].shape,item["pitch_yaw_roll_deg"])
        print("Input inspection passed.")
        return
    if not a.checkpoint:
        p.error("--checkpoint is required for inference")
    from adapters.panovggt import predict
    try:
        preds = predict([v["path"] for v in records],a.checkpoint,a.device)
        np.savez_compressed(out/"predictions.npz",**preds)
        meta["prediction_shapes"]={k:list(v.shape) for k,v in preds.items()}
        meta["status"]="inference_complete"
        save()
    except Exception as e:
        meta["status"]="failed";meta["error"]=str(e);save();raise
    print("Predictions saved:",out/"predictions.npz")

if __name__ == "__main__":
    main()
