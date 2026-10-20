"""Spherical diagnostics in the repository's endpoint-inclusive ERP convention."""
import numpy as np
from scipy.spatial.transform import Rotation
from geometry.poses import erp_rays, validate_poses


def unit(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim < 1 or vectors.shape[-1] != 3:
        raise ValueError("Expected (...,3) bearings")
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if not np.isfinite(vectors).all() or np.any(norms <= 0):
        raise ValueError("Bearings must be finite and nonzero")
    return vectors / norms


def angle_deg(a, b):
    """Stable angular separation, including zero and antipodal bearings."""
    a, b = unit(a), unit(b)
    return np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1),
                               np.sum(a * b, axis=-1)))


def spherical_grid(height, width):
    """Rays and exact solid-angle cells around the endpoint grid.

    Seam columns each receive a half-width cell. Polar rows receive cap
    cells, not zero weight. All weights sum to 4*pi steradians.
    """
    if not isinstance(height, int) or not isinstance(width, int) or min(height, width) < 2:
        raise ValueError("Integer dimensions >= 2 required")
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    lon = np.linspace(-np.pi, np.pi, width)
    lat = np.linspace(np.pi/2, -np.pi/2, height)
    lon_edges = np.r_[-np.pi, (lon[:-1] + lon[1:])/2, np.pi]
    lat_edges = np.r_[np.pi/2, (lat[:-1] + lat[1:])/2, -np.pi/2]
    area = (np.sin(lat_edges[:-1]) - np.sin(lat_edges[1:]))[:, None] * np.diff(lon_edges)
    return erp_rays(u, v, width, height), np.rad2deg(lon), np.rad2deg(lat), area


def bearing_coordinates(rays):
    rays = unit(rays)
    return (np.rad2deg(np.arctan2(rays[..., 0], rays[..., 2])),
            np.rad2deg(np.arcsin(np.clip(-rays[..., 1], -1, 1))))


def compensated_motion(a, b, pose_i, pose_j):
    """Matched rays; rotate camera-j bearings into camera-i, without fitting R.

    Residuals include translation, matching error, dynamics and pose error.
    They are not reconstruction errors or independently verified parallax.
    """
    validate_poses(np.stack([pose_i, pose_j]))
    a, b = unit(a), unit(b)
    if a.ndim != 2 or a.shape != b.shape:
        raise ValueError("Expected matching (N,3) bearing arrays")
    r_i_from_j = pose_i[:3, :3].T @ pose_j[:3, :3]
    compensated = b @ r_i_from_j.T
    return angle_deg(a, b), angle_deg(a, compensated)


def translation_parallax(rays, radial_range_m, pose_i, pose_j):
    """Analytical static-point parallax; caller supplies actual or synthetic range.

    No scene distance is inferred from camera poses alone.
    """
    validate_poses(np.stack([pose_i, pose_j]))
    rays = unit(rays)
    distance = np.asarray(radial_range_m, dtype=float)
    if not np.isfinite(distance).all() or np.any(distance <= 0):
        raise ValueError("Finite positive radial ranges required")
    delta_i = pose_i[:3, :3].T @ (pose_j[:3, 3] - pose_i[:3, 3])
    target_in_i = rays * np.broadcast_to(distance, rays.shape[:-1])[..., None] - delta_i
    return angle_deg(rays, target_in_i)


def select_pairs(poses, baseline_edges_m, per_bin=20, seed=17, max_rotation_deg=180.):
    """Uniform reservoir sample within [lo,hi) metric bins over all i<j pairs.

    O(N^2) distance work, O(N + bins*per_bin) memory. No temporal-gap proxy,
    nearest-bin substitution, or truncation to the earliest trajectory segment.
    """
    validate_poses(poses)
    edges = np.asarray(baseline_edges_m, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or edges[0] < 0 or np.any(np.diff(edges) <= 0):
        raise ValueError("Baseline edges must be finite, nonnegative and increasing")
    if isinstance(per_bin, bool) or not isinstance(per_bin, int) or per_bin < 1:
        raise ValueError("per_bin must be a positive integer")
    if not np.isfinite(max_rotation_deg) or not 0 <= max_rotation_deg <= 180:
        raise ValueError("max_rotation_deg must be in [0,180]")
    rng = np.random.default_rng(seed)
    pools = [[] for _ in edges[:-1]]
    counts = np.zeros(len(pools), dtype=np.int64)
    centres, rotations = poses[:, :3, 3], poses[:, :3, :3]
    for i in range(len(poses)-1):
        distances = np.linalg.norm(centres[i+1:] - centres[i], axis=1)
        bins = np.searchsorted(edges, distances, side="right") - 1
        candidates = np.flatnonzero((bins >= 0) & (bins < len(pools)))
        if not len(candidates):
            continue
        js = i + 1 + candidates
        angles = np.rad2deg(Rotation.from_matrix(rotations[i].T @ rotations[js]).magnitude())
        for k, j, theta in zip(candidates, js, angles):
            if theta > max_rotation_deg:
                continue
            b = int(bins[k])
            row = dict(bin=b, frame_i=i, frame_j=int(j),
                       baseline_m=float(distances[k]), rotation_deg=float(theta))
            counts[b] += 1
            if len(pools[b]) < per_bin:
                pools[b].append(row)
            else:
                slot = int(rng.integers(counts[b]))
                if slot < per_bin:
                    pools[b][slot] = row
    selected = sorted([row for pool in pools for row in pool],
                      key=lambda row: (row["bin"], row["frame_i"], row["frame_j"]))
    coverage = [dict(bin=b, lower_m=float(edges[b]), upper_exclusive_m=float(edges[b+1]),
                     eligible_pairs=int(counts[b]), selected_pairs=len(pools[b]))
                for b in range(len(pools))]
    return selected, coverage
