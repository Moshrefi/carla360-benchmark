# Ground-truth convention: source_matlab_v1

This profile is specific to the supplied `Maryam_drive.py`, `fisheyeomar.m`,
`fisheye_camera.py` and `FisheyeToSphere.m`. Changing the stitching rotation,
image flips, longitude origin, row/column indexing or sensor mount requires
a new profile. The exporter accepts the source labels as supplied and does
not infer capture synchronization or re-pair Front/Back frames.

## Source and output

Source CSV: headerless `x,y,z,pitch,yaw,roll`, metres and degrees. The index in
`equi_INDEX_x_y_z_pitch_yaw_roll.png` selects a **zero-based** CSV row. Only
images present in the input directory are exported, sorted by numeric index.
CSV values retain their full supplied float64 precision. Filename-only mode
uses rounded labels and records that limitation. No timestamps are invented.

World: right-handed `(CARLA X, -CARLA Y, CARLA Z)`, metres, same map origin.
Camera: right-handed `(right, down, forward)`. All transforms use column vectors.
`p_world = R_wc @ p_camera + t_wc`; `t_wc` is the camera centre in output world.
`T_i_from_j = inverse(T_wc_i) @ T_wc_j` maps camera j points into camera i.
The world origins of different CARLA maps are independent.

Quaternions are unit `(qx,qy,qz,qw)`, scalar last, canonical sign (`w>=0`).
At a 180-degree rotation, SciPy chooses a deterministic vector sign. Quaternions
can still have sign discontinuities; do not interpolate without sign handling.
CSV also includes all nine row-major rotation entries and native source poses.
`poses_c2w.npy` is float64 `(N,4,4)` in exactly the same order as `poses.csv`.

## Exact image sampling (no resampling)

The supplied MATLAB script uses `linspace(0,2*pi,W)` and `linspace(0,pi,H)`.
Images are copied **byte for byte**, so the exported convention preserves this
endpoint-inclusive grid, rather than silently substituting pixel centres:

```
lambda = 2*pi*u/(W-1) - pi
latitude = pi/2 - pi*v/(H-1)
d_camera = [cos(latitude)*sin(lambda), -sin(latitude), cos(latitude)*cos(lambda)]
```

`u,v` are zero-based pixel indices. Left and right edges share longitude;
top and bottom rows are poles. The continuous image midpoint faces +Z.
For even dimensions that midpoint lies between samples. A consumer expecting
pixel-centre sampling must explicitly resample or account for this grid.

## Derivation of the camera basis

The source's sphere is `s=[sin(theta)cos(phi),sin(theta)sin(phi),cos(theta)]`.
Let `d` be the camera ray defined above. Then `s = D @ d`, where:

```
D = [[0,0,-1], [-1,0,0], [0,-1,0]]
Q = [[1,0,0], [0,0,-1], [0,1,0]]       # normalized (w,x,y,z)=(1,1,0,0)
S = [[0,1,0], [1,0,0], [0,0,1]]        # MATLAB samples row=Xp,col=Yp
C = [[0,0,1], [1,0,0], [0,-1,0]]       # optical to native CARLA axes
B = C @ S @ Q @ D = diag(-1,1,1)
A = diag(1,-1,1)                         # CARLA world to output world
R_wc = A @ R_carla @ B
t_wc = A @ t_carla
```

Both A and B are reflections; their combination produces a proper rotation.
Do not pass either reflection alone to a quaternion conversion. The Back
hemisphere uses the opposite local view and its sign changes give the same
effective basis B. Tests independently trace both hemispheres through the
source projection/inverse-projection formulas, without pixel rounding.

CARLA's rotation matrix corresponds to `Rz(yaw) Ry(-pitch) Rx(-roll)`.
Reference: https://carla.org/Doxygen/html/de/d17/geom_2Transform_8h_source.html
MATLAB rotation API: https://www.mathworks.com/help/nav/ref/quat2rotm.html

The source quaternion is interpreted as normalized Rx(+90 degrees). To verify
the exact MATLAB environment used to generate a sequence, execute:

```matlab
R = quat2rotm(quaternion(1,1,0,0));
assert(norm(R - [1 0 0; 0 0 -1; 0 1 0], 'fro') < 1e-10);
```

## Validation and limits

`python -m unittest discover -s tests -v` checks cardinal directions, native
CARLA rotations, independent source-ray chains, gimbal-lock cases, relative
point transforms, full precision, numeric sorting, copy hashes and failure
handling. Export checks orthogonality, determinant +1 and quaternion roundtrip.

These validate conversion and packaging. They do not establish the accuracy
of the source labels or image/pose alignment. A scene-landmark reprojection
check has not been performed; this is explicitly recorded in metadata.

Output sequences are built in a temporary directory and published only when
complete. Existing sequence directories are refused, and source files are
never renamed or modified. Each sequence has its own metadata, optional map
name and CARLA version. Use one directory and one CSV per capture sequence.
