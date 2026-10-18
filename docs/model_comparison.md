# Four-model image-only comparison

Open `notebooks/06_model_depth_comparison.ipynb`. All new code and notebook text
are English. GT investigation is paused: no pose values, depth GT, calibration
correction, or scoring are used. The existing Standard_GT export remains intact.

## What is implemented

| Model | Input protocol | Depth | Temporal pose |
|---|---|---|---|
| PanoVGGT | Selected ERPs jointly, official 518x1036 loader | Native radial depth | Native predicted c2w |
| VGGT-360 | Each ERP separately, official adaptive views and fusion | Official norm-of-world-points fusion | No; saved poses describe virtual views of one panorama |
| OmniVGGT | Six perspective views per ERP, all selected ERPs jointly | Z-depth converted to radial using predicted intrinsics | Front virtual camera per ERP |
| MapAnything | Same perspective views as OmniVGGT, joint inference | Native depth_along_ray | Front virtual camera per ERP |

OmniVGGT/MapAnything use OUR six-view adaptation, not an official panoramic
benchmark protocol. Known crop orientations place the predicted depth pixels
back into ERP; the models do not receive those orientations or GT poses.
Predicted camera rays may disagree with crop geometry, causing residual spatial
inconsistency. Six virtual cameras share a physical centre, but the models are
not constrained to predict co-location. No scale fitting between faces is done.

VGGT-360 reproduces upstream's norm of predicted WORLD points before fusion.
This is a method-specific approximation, not a general conversion of arbitrary
world points into camera-centred radial depth. Its upstream sampling grid is
retained; small sampling differences from the viewer's endpoint grid are possible.

These differing input protocols must be reported if results are published.
MapAnything predicts metric scale; that is not evidence of metric accuracy on
CARLA. Other output scales are not assumed to be metres. No depth GT is currently
provided, so the notebook is for qualitative inspection, not accuracy rankings.

## 1. Download pinned source repositories

From the benchmark repository root:

```bash
git pull
git submodule update --init --recursive
python -m comparison.setup_sources
```

`sources.json` pins the three newly added upstream repositories. Existing
checkouts with different commits are preserved and reported as an error.
PanoVGGT uses the existing submodule; its actual commit is recorded per run.

## 2. Separate environments

Keep your working PanoVGGT environment. Set its environment name and checkpoint
path in the notebook. Do not install these dependencies over it.

Example setup for the new models (CUDA wheel below follows OmniVGGT's published
setup; your NVIDIA driver must support it). Run `nvidia-smi` first. If your server
requires a different CUDA wheel, use its matching PyTorch installation.

```bash
conda create -n vggt360 python=3.10 -y
conda run -n vggt360 python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
conda run -n vggt360 python -m pip install -r comparison/requirements-vggt360.txt

conda create -n omnivggt python=3.10 -y
conda run -n omnivggt python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
conda run -n omnivggt python -m pip install -r third_party/OmniVGGT/requirements.txt

conda create -n mapanything python=3.12 -y
conda run -n mapanything python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
conda run -n mapanything python -m pip install -e third_party/MapAnything
conda run -n mapanything python -m pip install scipy matplotlib
```

Example notebook environment:

```bash
conda create -n carla360-viewer python=3.10 -y
conda run -n carla360-viewer python -m pip install -r comparison/requirements-viewer.txt
conda run -n carla360-viewer python -m ipykernel install --user --name carla360-viewer --display-name "CARLA360 Comparison"
```

Select **CARLA360 Comparison** as the VS Code notebook kernel. Set the PanoVGGT
entry in `ENVIRONMENTS` to your existing model environment (for example `base`
only if that is where your working PanoVGGT installation actually lives).

Alternatively, for the notebook kernel, install `comparison/requirements-viewer.txt` in any
separate environment with Jupyter/VS Code kernel access. Inference runs in
subprocesses in the environments above; it does not load four models in the kernel.

VGGT-360 downloads `facebook/VGGT-1B`; MapAnything downloads
`facebook/map-anything`; OmniVGGT downloads `OmniVGGT.safetensors` from
`Livioni/OmniVGGT` unless a local checkpoint path is supplied. PanoVGGT requires
your existing local checkpoint. HF access/cache and sufficient disk are required.

## 3. Run the notebook

1. Set `SEQUENCE`, a new `JOB` folder, and `FRAME_IDS` (default two frames: 0,20).
2. Set conda environment names in `ENVIRONMENTS`; `None` uses the kernel's Python.
3. Set `CHECKPOINTS['panovggt']` to your existing file. Optional: a local OmniVGGT checkpoint.
4. Run preparation once. It saves a frozen manifest, RGB images and perspective crops.
5. Run each model cell. The models run sequentially and write separate results.
6. Run the plots; missing/failed models are visibly marked, never replaced with dummy outputs.

For a first GPU smoke run, set `ENABLED_MODELS = ['vggt360']`, then add the others.
The default enables all four. Two panoramas already mean twelve views for the
perspective models; no GPU memory requirement has been measured here. If a model
runs out of memory, use a NEW job with one panorama to test depth only, or use a
larger GPU. A one-frame run cannot establish a temporal trajectory.

CLI equivalent:

```bash
conda run --no-capture-output -n vggt360 python -m comparison.run --job outputs/depth_comparison/run01 --model vggt360
conda run --no-capture-output -n omnivggt python -m comparison.run --job outputs/depth_comparison/run01 --model omnivggt
conda run --no-capture-output -n mapanything python -m comparison.run --job outputs/depth_comparison/run01 --model mapanything
# Run this in your existing PanoVGGT environment:
python -m comparison.run --job outputs/depth_comparison/run01 --model panovggt --checkpoint /absolute/path/to/model.pt
```

Completed runs contain `predictions.npz` (radial_depth, frame_ids, optional
poses_c2w) and `metadata.json`. The latter is written last as a completion marker.
Partial runs are not plotted. To retry, move that model's failed output folder
aside; no existing output is silently overwritten. The manifest hash and frame
IDs must agree across all runs before plotting.

The normalized view uses independent per-image 2nd/98th percentiles and therefore
hides scale differences. Raw maps have explicit colorbars; they do not become
metric just because the same color range is selected. Missing depth is masked.
Trajectories are in each model's own first-camera frame and scale, without GT
alignment or accuracy metrics. A two-frame path is only a smoke-test visualization.

## Validation status

CPU tests cover orientation of virtual cameras, sphere z-to-radial conversion,
full ERP coverage, preservation of invalid depth, image-only manifest creation,
and refusal to overwrite a job. Python syntax and notebook cells are checked.
Model APIs were inspected against the pinned sources. GPU inference and checkpoint
loading have NOT been executed in the authoring environment; first server runs
are integration tests, not pre-verified benchmark results.

## Pinned VGGT-360 import fix

The published attention.py imports `per_image_confidence_minmem` from a missing
attention_utils.py, but never uses that symbol. Before importing the model,
the runner removes only that unused import from the local upstream checkout.
The original file SHA-256 is verified and the absence of symbol usages is checked
with Python AST. The operation is idempotent and refuses unexpected file contents.
The applied fix ID is recorded in successful run metadata. No replacement model
function or attention algorithm is introduced.

If a previous run failed during import, move its incomplete model output folder
aside before retrying (or choose a new JOB). The input preparation can be reused.
