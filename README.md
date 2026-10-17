# CARLA360 benchmark

Shared ERP dataset loader and model adapters. First integration: PanoVGGT.
Dataset acquisition is accepted as supplied; images are not altered or re-paired.

## Status
- Standard_GT export, unchanged image copies, explicit camera-to-world poses and a standard-sequence loader: implemented and tested.
- Dataset parsing, numeric frame order, full-precision CSV poses and RGB loading: implemented and smoke-tested.
- PanoVGGT adapter: implemented against pinned official source; GPU inference not yet run.
- GT pose metrics and other model adapters: not yet implemented. No benchmark scores claimed.

## Build Standard_GT (CPU, Python 3.10+)

From your local repository terminal in VS Code:

```sh
git pull --ff-only
python -m pip install -r requirements-gt.txt
```

Put the original `equi_*.png` ERP images and `positions_angles_camera1.csv`
directly in `dataset/`. The CSV has no header and uses `x,y,z,pitch,yaw,roll`.

```sh
python build_standard_gt.py --dataset dataset --pose-csv dataset/positions_angles_camera1.csv --output Standard_GT --sequence-id sequence001
```

The exporter reads full-precision poses from the CSV, sorts source frame IDs
numerically and copies images without changing their bytes. Original files stay
in place. An existing output sequence is never overwritten; choose a new
`--sequence-id` for another export. Optional `--map-name` and `--carla-version`
record capture provenance. Use one input folder and CSV per capture sequence.

```text
Standard_GT/sequence001/
  images/000000.png
  images/000001.png
  ...
  poses.csv
  poses_c2w.npy
  metadata.json
```

`poses.csv` contains new and original frame IDs/names, camera position in metres,
unit quaternion `(qx,qy,qz,qw)`, nine rotation-matrix entries, original source
poses and image hashes. `poses_c2w.npy` contains float64 camera-to-world matrices
in the same row order. Metadata documents the axes, image sampling and checks.

**Geometry profile:** this conversion is specific to the supplied MATLAB
stitching code. The images retain its endpoint-inclusive ERP sampling. World
axes are `(CARLA X, -CARLA Y, CARLA Z)` and camera axes are `(right, down, forward)`.
Read the derivation and assumptions in [docs/gt_convention.md](docs/gt_convention.md)
before using it for a differently generated sequence.

If `--pose-csv` is omitted, the exporter first looks for
`dataset/positions_angles_camera1.csv`; otherwise it uses rounded filename poses
and records that precision limitation. For the most accurate export, use the CSV.

For a step-by-step notebook, install `ipykernel`, select the project interpreter
in VS Code and open [notebooks/02_build_standard_gt.ipynb](notebooks/02_build_standard_gt.ipynb).
The callable function is `build_standard_gt.build_standard_gt(...)`.

```sh
python -m unittest discover -s tests -v
python run_eval.py --standard-sequence Standard_GT/sequence001 --inspect-only --output outputs/standard-inspect01
```

The same `--standard-sequence` input works with the PanoVGGT adapter below.
Conversion tests verify geometry formulas and file consistency; they do not
independently establish capture alignment. Model pose metrics remain a next step.

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
