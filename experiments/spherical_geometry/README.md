# Spherical geometry research benchmark — Phase 1

## Audit and implementation plan

Audited main commit 1a931e15881f583fd1c657faae7da8972835c91f before changes.

- Standard_GT uses metadata.json, poses.csv and float64 poses_c2w.npy, with
  byte-identical ERP images. Reuse datasets.standard_gt.StandardGTDataset through
  analysis.dataset_stats.analyze_sequence; no new filename/CSV pose parser.
- geometry.poses provides validated proper camera-to-world transformations and
  endpoint-inclusive ERP rays. World coordinates are (CARLA X, -CARLA Y, CARLA Z)
  in metres; camera coordinates are right/down/forward. Reuse erp_rays.
- analysis.dataset_stats already computes adjacent translation, SO(3) rotation,
  Euler coverage, retained path length and source-ID gaps. Reuse its exports.
- Notebook 03 covers trajectory statistics. Notebook 05 and
  diagnostics.rotation_check explicitly investigate possible camera-basis and
  source-row offset errors. Conversion correctness does not establish image/pose
  alignment. No corrections are applied here.
- Notebooks 08/09 use comparison.bench_ui and isolated bench_infer workers.
  Their selection is file-order/stride based, and chunks have independent scales.
  The new module never invokes or modifies these inference pipelines.
- VGGT uses upstream default preprocessing; Omega uses image_resolution;
  OmniVGGT uses target_size and ImageNet normalization; PanoVGGT uses fixed
  518x1036 ERP preprocessing. The first three expose native perspective z-depth
  on direct ERP; this must not be relabeled spherical metric range.
- Repository source has no committed dataset, depth GT, or completed research
  measurements. Full sequence and GPU execution require the user's Rails data.

Implementation sequence: (1) reuse pose statistics, (2) metric pair selection,
(3) spherical cells and compensated bearings, (4) image diagnostics and exports,
(5) analytical tests plus notebook integration, (6) PR review. Phase 2 is deferred.

## Run

Use Python 3.11 in a dedicated CPU environment:

    python -m pip install -r experiments/spherical_geometry/requirements.txt
    python -m unittest discover -s tests -v
    python experiments/spherical_geometry/validate_notebook.py

Open notebooks/10_spherical_geometry_phase1.ipynb, or edit a copy of
example_config.json and run:

    python -m experiments.spherical_geometry.run --config experiments/spherical_geometry/example_config.json

Point sequence at an existing Standard_GT sequence. Every run requires a new
output path, outside the input sequence. Default outputs/ is already gitignored.
No checkpoints or datasets are added or modified. Tests generate tiny temporary
synthetic fixtures; they are not evaluation results.

## Geometry and sampling

For camera-to-world poses (R_i,C_i), baseline is ||C_j-C_i|| in metres.
Relative orientation magnitude is the SO(3) angle of R_i.T R_j.
Matched camera-j bearings are rotated into camera-i with R_i.T R_j.
Angular residual uses atan2(||a cross b||, a dot b), stable near zero and pi.

All i<j pairs are considered. Bins are [lower,upper), including zero baseline,
excluding the last upper edge. A fixed-seed reservoir samples uniformly per bin
after applying the configured rotation cap. Empty bins are never filled with
out-of-range pairs. O(N^2) time, O(N + bins*quota) memory; do not use on millions
of frames without a spatial-index extension. All-pair eligible counts and
baseline-stratified selected distributions are labeled separately.
Diagnostic pairs are subsampled per bin with fixed seed 17, recorded in provenance.

ERP rows and columns follow the repository's endpoint grid. Integration-cell
boundaries lie halfway between samples. Longitude seam columns have half cells;
polar rows have cap cells. Areas sum to 4*pi. These weights are for dense grid
quantities, not a correction for nonuniform feature detection.

translation_parallax is an analytical utility requiring a supplied radial range.
For a point directly ahead at range D and lateral baseline B, the angle is
atan(B/D). It does not infer scene distance or fabricate a depth map.

## Diagnostic interpretation

Image features and mutual SIFT matches reuse diagnostics.rotation_check.
Known poses compensate rotation; no image-fitted rotation or residual cutoff
removes translation. Every tentative match is exported. Pair summaries include
match counts and occupied longitude sectors; status "measured" means residuals
were computed, not that the matches or GT passed physical validation.

The existing SIFT extractor excludes the top/bottom 10% and seam margin.
Sparse maps never fill missing pixels. Latitude profiles pool feature samples,
so pairs with more features contribute more. They must not be described as
uniform-sphere error profiles. No geometric outlier rejection is applied:
moving objects, false matches, stitch artifacts and pose errors can inflate
residuals. Inspect spatial coverage and notebook 05 before interpreting them.

Without calibrated depth or independently verified correspondences, scene
distance is unknown and a residual is not proven translation parallax.
No depth errors, model rankings, velocity or timing estimates are reported.

## Outputs

- config.json, provenance.json, run_status.json: settings, package/code/input
  hashes, pose assumptions and explicit completion/export status.
- trajectory_statistics/: existing per-frame/per-step CSV and summary JSON.
- selected_pairs.csv, baseline_coverage.csv, frame_manifest.json: reproducible
  pair identities, exact eligibility counts, actual verified selected-image hashes
  and camera-to-world poses for later inference reuse.
- spherical_rows.csv: latitude and integrated row solid angle.
- parallax_matches.csv, parallax_pairs.csv, latitude_residuals.csv: actual
  tentative-match residuals and their coverage. No latitude table when no matches.
- figures/: self-contained Plotly HTML, PNG at 3x resolution and PDF.
  figure_exports.json records each static export failure; the CLI returns a
  nonzero exit after preserving tables and HTML if a static export fails.
  Plotly 3D PDF traces may be rasterized; 2D curve traces are vector graphics.

The notebook is committed without outputs. Raw run data and figures belong in
outputs/, never in Git. A failed run retains status "started" for inspection;
do not treat it as complete or reuse the directory.

## Phase 2 review gate

Not implemented. After Phase 1 validation and PR review:
1. Validate image/pose alignment and confirm scene-distance/depth availability.
2. Feed identical pair manifests to existing VGGT, VGGT-Omega, PanoVGGT and
   OmniVGGT workers. Record resize/crop/pad, angular coverage, sampling grid,
   model revision, checkpoint hash and joint context.
3. Establish native camera axes, c2w/w2c, z-depth versus range, and the mapping
   back to original ERP pixels. Do not convert perspective z-depth on direct
   ERP with an unjustified spherical unprojection.
4. Report native metric-scale results only for verified metric predictions.
   Report relative-scale evaluation separately, with explicit alignment type,
   fitted data subset and degeneracy checks. Never silently align each image.
5. Compute depth and latitude error profiles only with valid, co-registered GT,
   validity masks and spherical-area weighting. Unknown GT means unavailable.
6. Add actual scene-distance strata, ERP depth maps and model comparisons.

## Regression validation

CI runs the new analytical/integration tests independently, then runs the full
suite on both the audited base and the branch. New failures or changed skips
fail CI. Existing base failures are printed explicitly rather than silently
skipped. The baseline test named test_argus_never_launches_process predates the
Argus integration and still expects pending behavior; it may fail on main.
The baseline comparison checks this empirically and does not change that pipeline.
