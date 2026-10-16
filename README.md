# CARLA360 benchmark

Shared ERP dataset loader and model adapters. First integration: PanoVGGT.
Dataset acquisition is accepted as supplied; images are not altered or re-paired.

## Status
- Dataset parsing, numeric frame order, full-precision CSV poses and RGB loading: implemented and smoke-tested.
- PanoVGGT adapter: implemented against pinned official source; GPU inference not yet run.
- GT pose metrics and other model adapters: not yet implemented. No benchmark scores claimed.

## Open in VS Code (Windows PowerShell or Ubuntu)

After cloning your repository with `git clone --recurse-submodules YOUR_REPO_URL`, open its folder in VS Code.
Install the Microsoft Python and Jupyter extensions. Create an environment:

```sh
python -m venv .venv
```

Windows PowerShell: `.venv\Scripts\Activate.ps1`
Ubuntu: `source .venv/bin/activate`

```sh
python -m pip install -r requirements.txt
python -m ipykernel install --user --name carla360 --display-name "Python (carla360)"
git submodule update --init --recursive
```

Select this interpreter/kernel in VS Code. Open `notebooks/01_inspect_and_run.ipynb` and run one cell at a time.
The data inspection runs on CPU. GPU inference can run on an Ubuntu server through VS Code Remote SSH.

## Your data (kept outside Git)

```text
data/carla360/
  positions_angles_camera1.csv
  images/equi_3_-158.227_-79.992_75.000_74.280_26.899_89.962.png
```

CSV has no header: x,y,z,pitch,yaw,roll. Filename index selects the zero-based CSV row.
Filename values must agree with that row to three decimal places. Frames may be a subset.

```sh
python run_eval.py --image-dir data/carla360/images --pose-csv data/carla360/positions_angles_camera1.csv --inspect-only --output outputs/inspect01
```

## PanoVGGT inference

Use a separate environment per model when dependencies conflict. Install CUDA-enabled PyTorch appropriate to your GPU following https://pytorch.org/get-started/locally/ and the pinned upstream README, then:

```sh
python -m pip install -r third_party/PanoVGGT/requirements.txt
```

Download the official checkpoint from https://huggingface.co/YijingGuo/PanoVGGT/resolve/main/model.pt to `checkpoints/panovggt.pt`.
Run a short sequence first (all selected frames are processed jointly):

```sh
python run_eval.py --image-dir data/carla360/images --pose-csv data/carla360/positions_angles_camera1.csv --checkpoint checkpoints/panovggt.pt --limit 3 --output outputs/panovggt01
```

Outputs: `manifest.json` (frame IDs, source paths, GT, upstream commit, status) and raw `predictions.npz` (including available depth/poses/points). GT is never passed into the model.
Official preprocessing resizes ERP to 1036x518. Depth units are not assumed metric.
Different runs use separate empty output directories. No full trajectory concatenation across independent windows is performed.

## Updating in VS Code

Use `git pull` followed by `git submodule update --init --recursive` to receive committed updates. Changes made here are not streamed live into your local editor.

## Next milestones

1. Run the three-frame GPU smoke test.
2. Confirm output pose conventions and implement rotation/trajectory metrics with synthetic checks.
3. Add VGGT and Pi3 adapters with explicit perspective-view transforms.

Upstream source keeps its own license in `third_party/PanoVGGT/LICENSE`.
