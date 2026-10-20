"""CPU-only Phase 1 runner. No model weights, inference, or depth evaluation."""
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from analysis.dataset_stats import analyze_sequence, export_statistics
from comparison.bench_ui import digest
from .geometry import select_pairs, spherical_grid, compensated_motion, bearing_coordinates
from .plots import pose_figures, erp_map, motion_figures, save_figures

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    sequence: str
    output: str
    baseline_edges_m: tuple = (0., .01, .1, .5, 1., 2., 5., 10.)
    pairs_per_bin: int = 10
    seed: int = 17
    max_rotation_deg: float = 180.
    image_diagnostics: bool = True
    diagnostic_pairs_per_bin: int = 2
    latitude_bin_width_deg: float = 15.
    grid_height: int = 181
    grid_width: int = 361
    static_formats: tuple = ("png", "pdf")


def git_info():
    def command(*args):
        p = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    try:
        return {"commit": command("rev-parse", "HEAD"), "working_tree": command("status", "--porcelain")}
    except OSError:
        return {"commit": None, "working_tree": None}


def diagnose(ds, selected, per_bin, latitude_width):
    from diagnostics.rotation_check import image_features, matched_bearings
    import cv2
    cv2.setNumThreads(1)
    cache = lru_cache(maxsize=8)(lambda i: image_features(ds.paths[i]))
    summaries, measurements = [], []
    taken = {}
    # Independent seeded subsample of the baseline-selected pairs; avoid an early-route bias.
    rng = np.random.default_rng(17)
    for index in rng.permutation(len(selected)):
        p = selected[index]
        if taken.get(p["bin"], 0) >= per_bin:
            continue
        taken[p["bin"]] = taken.get(p["bin"], 0) + 1
        i, j = p["frame_i"], p["frame_j"]
        pair_id = f"{i}_{j}"
        row = dict(p, pair_id=pair_id)
        try:
            a, b, uv1, uv2, hw1, hw2 = matched_bearings(cache(i), cache(j))
            if not len(a):
                raise ValueError("No mutual ratio-test matches")
            raw, residual = compensated_motion(a, b, ds.poses[i], ds.poses[j])
            lon, lat = bearing_coordinates(a)
            sectors = np.unique(np.minimum(7, ((lon+180)/45).astype(int)))
            row.update(status="measured", matches=len(a), longitude_sectors=len(sectors),
                       median_residual_deg=float(np.median(residual)),
                       p95_residual_deg=float(np.percentile(residual, 95)),
                       median_raw_motion_deg=float(np.median(raw)))
            for k in range(len(a)):
                measurements.append(dict(pair_id=pair_id, frame_i=i, frame_j=j,
                    u_i=float(uv1[k, 0]), v_i=float(uv1[k, 1]),
                    u_j=float(uv2[k, 0]), v_j=float(uv2[k, 1]),
                    longitude_deg=float(lon[k]), latitude_deg=float(lat[k]),
                    latitude_bin=int(np.clip(np.floor((lat[k]+90)/latitude_width),
                                             0, int(np.ceil(180/latitude_width))-1)),
                    raw_motion_deg=float(raw[k]), residual_deg=float(residual[k])))
        except (ValueError, cv2.error) as exc:
            row.update(status="inconclusive", reason=str(exc), matches=0)
        summaries.append(row)
    return pd.DataFrame(summaries), pd.DataFrame(measurements)


def run(config):
    if not isinstance(config, Config):
        raise TypeError("Expected Config")
    if (not isinstance(config.diagnostic_pairs_per_bin, int) or config.diagnostic_pairs_per_bin < 1
            or not np.isfinite(config.latitude_bin_width_deg)
            or not 0 < config.latitude_bin_width_deg <= 180):
        raise ValueError("Invalid diagnostic sample count or latitude bin width")
    if any(fmt not in ("png", "pdf") for fmt in config.static_formats):
        raise ValueError("Static format must be png or pdf")
    result = analyze_sequence(config.sequence)
    ds = result["dataset"]
    if (ds.metadata.get("geometry_profile") != "source_matlab_v1"
            or ds.metadata.get("world", {}).get("unit") != "metre"
            or ds.metadata.get("erp", {}).get("sampling") != "endpoint_inclusive"):
        raise ValueError("Requires source_matlab_v1, metres, endpoint-inclusive ERP")
    selected, coverage = select_pairs(ds.poses, config.baseline_edges_m,
        config.pairs_per_bin, config.seed, config.max_rotation_deg)
    _, lon, lat, area = spherical_grid(config.grid_height, config.grid_width)
    out = Path(config.output).expanduser().resolve()
    if out == ds.root or out.is_relative_to(ds.root):
        raise ValueError("Outputs must be outside the input sequence")
    out.mkdir(parents=True, exist_ok=False)
    write_json = lambda name, obj: (out/name).write_text(
        json.dumps(obj, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    write_json("config.json", asdict(config))
    write_json("run_status.json", {"status": "started"})
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), **git_info(),
                  "python": sys.version, "metadata": ds.metadata,
                  "input_files": {p: digest(ds.root/p) for p in ("poses.csv", "poses_c2w.npy", "metadata.json")},
                  "versions": {p: importlib.metadata.version(p) for p in
                               ("numpy", "scipy", "pandas", "Pillow", "plotly")},
                  "source_sha256": {str(p.relative_to(ROOT)): digest(p)
                      for folder in ("experiments/spherical_geometry", "geometry", "datasets", "analysis", "diagnostics")
                      for p in sorted((ROOT/folder).glob("*.py"))},
                  "depth_gt_status": "not_loaded_or_evaluated",
                  "capture_alignment": "not_independently_validated",
                  "diagnostic_sampling_seed": 17,
                  "limitations": [
                      "Residuals combine translation, matching error, dynamics, stitching and pose errors.",
                      "SIFT excludes seam and polar regions; missing coverage is not zero error.",
                      "Feature profiles are feature-weighted, not uniform-sphere error estimates.",
                      "Scene distance cannot be measured from poses alone.",
                      "No timestamps: translation and rotation per retained step, not velocities.",
                      "No model evaluation or scale alignment in Phase 1."]}
    write_json("provenance.json", provenance)
    export_statistics(result, out/"trajectory_statistics")
    for p in selected:
        for key in ("i", "j"):
            idx = p["frame_"+key]
            p["source_id_"+key] = int(ds.rows[idx]["source_frame_id"])
    pair_columns = ["bin", "frame_i", "frame_j", "baseline_m", "rotation_deg", "source_id_i", "source_id_j"]
    pairs = pd.DataFrame(selected, columns=pair_columns)
    pairs.to_csv(out/"selected_pairs.csv", index=False)
    pd.DataFrame(coverage).to_csv(out/"baseline_coverage.csv", index=False)
    # Freeze exact frame identities for later reuse by existing inference orchestration.
    frames = sorted({p["frame_"+key] for p in selected for key in ("i", "j")})
    records = []
    for idx in frames:
        actual = digest(ds.paths[idx])
        if actual != ds.rows[idx]["image_sha256"]:
            raise ValueError(f"Image hash mismatch at frame {idx}")
        records.append({"frame_id": idx, "source_frame_id": int(ds.rows[idx]["source_frame_id"]),
                        "path": str(ds.paths[idx]), "sha256": actual,
                        "pose_c2w": ds.poses[idx].tolist()})
    write_json("frame_manifest.json", records)
    spherical = pd.DataFrame({"latitude_deg": lat, "row_solid_angle_sr": area.sum(axis=1),
                              "fraction_of_sphere": area.sum(axis=1)/(4*np.pi)})
    spherical.to_csv(out/"spherical_rows.csv", index=False)
    figures = pose_figures(result, pairs, coverage)
    figures["erp_solid_angle"] = erp_map(area, lon, lat, "ERP sampling cell solid angle", "sr")
    summaries, matches = pd.DataFrame(), pd.DataFrame()
    if config.image_diagnostics:
        summaries, matches = diagnose(ds, selected, config.diagnostic_pairs_per_bin, config.latitude_bin_width_deg)
        summaries.to_csv(out/"parallax_pairs.csv", index=False)
        matches.to_csv(out/"parallax_matches.csv", index=False)
        if not matches.empty:
            matches.groupby("latitude_bin").agg(
                latitude_mean_deg=("latitude_deg", "mean"), count=("residual_deg", "size"),
                median_residual_deg=("residual_deg", "median")).to_csv(out/"latitude_residuals.csv")
        figures.update(motion_figures(matches, summaries))
    statuses = save_figures(figures, out/"figures", config.static_formats)
    write_json("figure_exports.json", statuses)
    failed = [s for s in statuses if s["status"] == "failed"]
    write_json("run_status.json", {"status": "completed_with_export_failures" if failed else "completed",
                                  "static_export_failures": len(failed),
                                  "selected_pairs": len(selected), "image_diagnostics": config.image_diagnostics})
    return {"output": out, "statistics": result, "pairs": pairs, "coverage": coverage,
            "matches": matches, "diagnostics": summaries, "figures": figures,
            "figure_exports": statuses}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON object matching Config")
    args = parser.parse_args()
    cfg = Config(**json.loads(Path(args.config).read_text()))
    result = run(cfg)
    print("Saved:", result["output"])
    if any(s["status"] == "failed" for s in result["figure_exports"]):
        raise SystemExit("Static export failed; inspect figure_exports.json. Tables and HTML were saved.")


if __name__ == "__main__":
    main()
