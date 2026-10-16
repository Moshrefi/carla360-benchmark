"""Use the pinned official image loading and inference functions; GT is never input."""
from pathlib import Path
import importlib.util
import sys
import numpy as np

def predict(image_paths, checkpoint, device="cuda"):
    root = Path(__file__).resolve().parents[1] / "third_party" / "PanoVGGT"
    script = root / "inference.py"
    if not script.exists():
        raise RuntimeError("Run git submodule update --init --recursive first")
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("official_panovggt_inference", script)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select --device cpu explicitly")
    model = upstream.load_model(str(root/"training/config/default.yaml"), str(Path(checkpoint).resolve()), device)
    model.eval().to(device)
    result = upstream.run_inference(model, image_paths, device)
    return {k:v for k,v in result.items() if isinstance(v, np.ndarray)}
