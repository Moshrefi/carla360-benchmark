# Read-only GT rotation diagnostic

Run `python -m pip install -r requirements-diagnostics.txt`, then open
`notebooks/05_gt_rotation_diagnostic.ipynb` in VS Code using that environment.
Set `SEQUENCE` to the exported Standard_GT sequence and `SOURCE_CSV` to the
original headerless capture CSV (x,y,z,pitch,yaw,roll), not poses.csv.

The diagnostic is specific to source_matlab_v1. It verifies the original CSV
against every exported source pose before checking images. Source rows are
zero-based and taken from source_csv_row, falling back to source_frame_id.
Deleted images never cause retained list indices to replace source indices.

Fixed candidates are current GT, original CSV row minus one, candidate ERP
camera basis, and the combination. The basis was inferred from earlier supplied
samples, not fitted anew to each evaluated pair. Source IDs 0–5 are excluded.
Triples must have consecutive original IDs/CSV rows and a common recorded centre
within 1 mm under both row hypotheses. Two disjoint triples per available region
(beginning 0–20%, middle 40–60%, end 80–100%) are sampled by default.

Mutual SIFT matches use ratio 0.7. Spherical RANSAC uses a 0.6 degree threshold,
70% training matches and 30% held-out matches. Minimum consensus: 25 training
and 10 held-out inliers, each at least 35% of its split; coverage in at least
three of eight longitude sectors is required. Independently measurable panorama
halves must agree within one degree. Missing half measurements are not proof
of consistency. These thresholds are diagnostic heuristics, not confidence bounds.

Reports contain full SO(3) geodesic rotation errors in degrees, not merely the
difference in scalar turn angles. Correlated pairs within one triple are not
independent samples. Low feature quality or missing regions is inconclusive.
No GT or image is modified; reports are written to a new timestamped outputs
folder. The ZIP contains JSON, CSV and (when fits pass) a plot.

This checks relative rotation only. It does not establish translation accuracy,
absolute camera calibration, unique timing offsets or full-dataset correctness.
Same-centre metadata does not independently establish actual capture centres.
Repeated angle schedules may make the apparent one-row offset non-unique.
