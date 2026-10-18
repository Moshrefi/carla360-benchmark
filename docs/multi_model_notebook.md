# Notebook 08: direct ERP experiments

Open `notebooks/08_multi_model_benchmark.ipynb` in VS Code Remote on Rails.
Keep the working notebook kernel. All inference subprocesses use
`~/.conda/envs/carla360-models/bin/python`, configurable in Step 1. No installs
or environment changes are performed. The viewer needs NumPy, Pillow,
Matplotlib, pandas and IPython in the selected notebook kernel.

1. Start with ACTION="preview" and the existing datasets/test_001 input folder.
2. Check selected files, model list and chunk count.
3. Set ACTION="run", then execute numbered steps in order.
4. Use ACTION="view" to display the saved run without GPU inference.
5. For more images, set INPUT_DIR="datasets/images", LIMIT=None (or a cap),
   START/STOP/STRIDE and CHUNK_SIZE. Use a new RUN_NAME for a new selection.

STOP is exclusive and positions are in the sorted file list, not source IDs.
Use natural sorting for numeric names or lexical sorting to reproduce prior
manual scripts. Chunk size defines joint model context. Independent chunks
have independent coordinate frames/scales; paths are never concatenated.
Each model/chunk reloads weights: suitable for controlled experiments, not an
optimized full-dataset throughput implementation. GPU OOM is recorded; reduce
chunk size in a new run. No automatic precision/context changes occur.

## Protocol and provenance

VGGT and Omega reuse the exact model constructors, loaders and camera decoders
supplied by the user. Omega uses image_resolution=416. OmniVGGT reuses the
supplied successful direct-ERP image-only invocation with BF16 autocast and
empty camera/depth GT indices. This differs intentionally from notebook 07's
six-face adaptation. PanoVGGT uses official fixed-size preprocessing and model
configuration, but strict checkpoint loading replaces the demo's strict=False.
A strict mismatch is an error; no missing model parameters are accepted.

VGGT-360 uses the official adaptive view generation and attention fusion with
a local VGGT checkpoint loaded strictly. That local checkpoint wrapper has not
been GPU-tested here. The exact known unused-import repair from notebook 07 is
reused; any other source variant is rejected by that repair. Its world-point
norm fusion is retained, not silently reinterpreted as general radial depth.
Argus is visible as pending and never loads substitute weights.

All runs preserve raw Float32 outputs. Native depth semantics differ; there is
no cross-model metric depth score. Normalized plots hide scale and raw plots
are explicitly nonmetric. Input hashes, checkpoint hash, source Git revision,
settings, precision, output shape/finite statistics and integrity hash are
saved. Cache request identity also hashes Python/YAML source and worker code.
New run outputs are stored separately from all earlier manual outputs.
Legacy files are inventoried but are not treated as a verified new cache when
image hashes/settings are absent. PNG previews are never imported as raw depth.

GT conditioning, GT evaluation, alignment and pose ranking are disabled.
Native points are displayed only where available; no fabricated point clouds
or pinhole unprojection of ERP depth is used. Trajectory panels show each
model separately with native axes and scale, one chunk at a time.

## Validation, 2026-10-18

Actually executed here on CPU:
- All helper Python and notebook code cells compile.
- Natural selection, slicing/stride, chunk partition [2,2,1] and exact copied
  input hashes checked on synthetic images.
- Changed selection/chunk boundaries and mutated inputs rejected.
- Cache settings mismatch and output corruption rejected.
- Depth panels (including pending Argus), native pose and point plots exercised
  using explicitly synthetic arrays; these are not model predictions.
- Argus selection verified to launch no inference process.

Command: `python -m unittest discover -s tests -p test_notebook_benchmark.py -v`

GPU inference in this new wrapper has NOT been executed by the author.
The user's successful manual runs establish the starting interfaces, not
validation of this notebook. On Rails validate the default three-image run,
check strict loading, inspect logs and metadata, then increase data selection.
The current authoring environment has no access to Rails weights/results.
