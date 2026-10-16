"""ERP dataset. CSV columns: x,y,z,pitch,yaw,roll (degrees), no header."""
from pathlib import Path
import numpy as np
from PIL import Image

class Carla360Dataset:
    def __init__(self, image_dir, pose_csv, indices=None):
        self.poses = np.loadtxt(pose_csv, delimiter=",", ndmin=2)
        if self.poses.shape[1] != 6 or not np.isfinite(self.poses).all():
            raise ValueError("Expected six finite CSV columns: x,y,z,pitch,yaw,roll")
        self.records = []
        seen = set()
        for path in Path(image_dir).glob("equi_*.png"):
            fields = path.stem.split("_")
            if len(fields) != 8:
                raise ValueError(f"Unexpected ERP filename: {path.name}")
            idx = int(fields[1])
            if indices is not None and idx not in indices:
                continue
            if idx in seen or idx < 0 or idx >= len(self.poses):
                raise ValueError(f"Duplicate or out-of-range frame index: {idx}")
            seen.add(idx)
            # Filenames are rounded to three decimals; CSV is authoritative.
            if not np.allclose(np.array(fields[2:], float), self.poses[idx], atol=0.00051, rtol=0):
                raise ValueError(f"Filename/CSV mismatch: {path.name}")
            self.records.append((idx, path))
        self.records.sort(key=lambda p: p[0])
        if not self.records:
            raise ValueError("No matching equi_*.png images")
    def __len__(self):
        return len(self.records)
    def __getitem__(self, item):
        idx, path = self.records[item]
        with Image.open(path) as im:
            rgb = np.array(im.convert("RGB"))
        if rgb.shape[1] != 2 * rgb.shape[0]:
            raise ValueError(f"Expected 2:1 ERP: {path.name}")
        return {"frame_id": idx, "path": str(path.resolve()), "rgb": rgb,
                "position_carla": self.poses[idx,:3].copy(),
                "pitch_yaw_roll_deg": self.poses[idx,3:].copy()}
    def manifest(self):
        return [{"frame_id": i, "path": str(p.resolve()),
                 "pose_csv": self.poses[i].tolist()} for i,p in self.records]
