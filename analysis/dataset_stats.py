"""Statistics over retained frames; no timestamps or ground heights are assumed."""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from datasets.standard_gt import StandardGTDataset


def describe(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return dict.fromkeys(['count', 'min', 'p05', 'median', 'mean', 'p95', 'max', 'std'], None) | {'count': 0}
    return {'count': int(values.size), 'min': float(values.min()),
            'p05': float(np.percentile(values, 5)), 'median': float(np.median(values)),
            'mean': float(values.mean()), 'p95': float(np.percentile(values, 95)),
            'max': float(values.max()), 'std': float(values.std())}


def circular_span_deg(angles):
    """Length of the smallest circular arc covering scalar angles, in [0,360)."""
    a = np.sort(np.mod(np.asarray(angles, dtype=float), 360))
    if not a.size:
        return None
    return float(360 - np.diff(np.r_[a, a[0] + 360]).max())


def pose_metrics(poses, source_ids, source_euler_deg, translation_threshold_m=0.01,
                 rotation_threshold_deg=0.1):
    from geometry.poses import validate_poses
    poses = np.asarray(poses, dtype=float)
    validate_poses(poses)
    ids = np.asarray(source_ids, dtype=np.int64)
    euler = np.asarray(source_euler_deg, dtype=float)
    n = len(poses)
    if ids.shape != (n,) or euler.shape != (n, 3) or not np.isfinite(euler).all():
        raise ValueError('Expected N source IDs and finite (N,3) pitch,yaw,roll')
    if np.any(np.diff(ids) <= 0):
        raise ValueError('Source frame IDs must be strictly increasing within one sequence')
    thresholds = np.array([translation_threshold_m, rotation_threshold_deg])
    if not np.isfinite(thresholds).all() or np.any(thresholds < 0):
        raise ValueError('Thresholds must be finite and nonnegative')
    xyz = poses[:, :3, 3]
    r = poses[:, :3, :3]
    delta = np.diff(xyz, axis=0)
    distance = np.linalg.norm(delta, axis=1)
    # Magnitude of the relative SO(3) rotation, stable near 0 and 180 degrees.
    rel = r[:-1].transpose(0, 2, 1) @ r[1:]
    rotation = np.rad2deg(Rotation.from_matrix(rel).magnitude()) if n > 1 else np.empty(0)
    from_first = np.rad2deg(Rotation.from_matrix(r[0].T @ r).magnitude())
    gap = np.diff(ids)
    low_translation = distance <= translation_threshold_m
    high_rotation = rotation > rotation_threshold_deg
    angle_ranges = {}
    for j, name in enumerate(['pitch', 'yaw', 'roll']):
        angle_ranges[name] = {
            'source_min_deg': float(euler[:, j].min()),
            'source_max_deg': float(euler[:, j].max()),
            'raw_span_deg': float(np.ptp(euler[:, j])),
            'circular_span_deg': circular_span_deg(euler[:, j]),
        }
    path_length = float(distance.sum())
    summary = {
        'frame_count': n, 'pair_count': n-1,
        'retained_path_length_m': path_length,
        'start_to_end_distance_m': float(np.linalg.norm(xyz[-1]-xyz[0])),
        'xyz_extent_m': np.ptp(xyz, axis=0).tolist(),
        'world_z_m': describe(xyz[:, 2]),
        'step_distance_m': describe(distance),
        'adjacent_rotation_deg': describe(rotation),
        'rotation_from_first_deg': describe(from_first),
        'source_euler_ranges': angle_ranges,
        'source_id_gaps': {'pair_count': int(np.sum(gap > 1)),
                           'missing_ids_between_retained_frames': int(np.maximum(gap-1, 0).sum())},
        'thresholds': {'translation_m': float(translation_threshold_m),
                       'rotation_deg': float(rotation_threshold_deg)},
        'low_translation_pair_count': int(low_translation.sum()),
        'low_translation_high_rotation_pair_count': int(np.sum(low_translation & high_rotation)),
        'consecutive_source_id_step_distance_m': describe(distance[gap == 1]),
        'consecutive_source_id_rotation_deg': describe(rotation[gap == 1]),
        'gapped_source_id_step_distance_m': describe(distance[gap > 1]),
    }
    return {'summary': summary, 'xyz': xyz, 'euler_deg': euler, 'source_ids': ids,
            'delta_xyz_m': delta, 'step_distance_m': distance, 'rotation_step_deg': rotation,
            'rotation_from_first_deg': from_first, 'source_id_delta': gap,
            'cumulative_distance_m': np.r_[0, np.cumsum(distance)]}


def analyze_sequence(sequence_dir, translation_threshold_m=0.01, rotation_threshold_deg=0.1):
    ds = StandardGTDataset(sequence_dir)  # Does not decode images during statistics.
    ids = [int(row['source_frame_id']) for row in ds.rows]
    angles = [[float(row['source_'+k+'_deg']) for k in ['pitch', 'yaw', 'roll']] for row in ds.rows]
    result = pose_metrics(ds.poses, ids, angles, translation_threshold_m, rotation_threshold_deg)
    result['dataset'] = ds
    result['summary'].update({
        'sequence_id': ds.metadata['sequence_id'], 'map_name': ds.metadata.get('map_name'),
        'geometry_profile': ds.metadata['geometry_profile'], 'pose_source': ds.metadata['pose_source'],
        'image_sizes_wh': sorted({(int(r['width']), int(r['height'])) for r in ds.rows}),
        'notes': [
            'Pairs refer to adjacent retained frames in source-ID order, including gaps.',
            'Path length is the sum of straight segments between retained camera centres.',
            'World Z is not height above terrain. No speed/FPS/duration is inferred.',
            'Euler ranges describe source CARLA angles; SO(3) magnitudes describe orientation change.',
            'Circular spans handle wraparound, but Euler components remain convention dependent.',
            'Missing IDs mean absent indices inside the observed range; their cause is unknown.',
        ],
    })
    return result


def comparison_row(result):
    s = result['summary']
    return {'sequence': s.get('sequence_id', ''), 'frames': s['frame_count'],
            'retained_path_m': s['retained_path_length_m'],
            'world_z_range_m': s['xyz_extent_m'][2],
            'step_median_cm': None if s['step_distance_m']['median'] is None else 100*s['step_distance_m']['median'],
            'step_p95_cm': None if s['step_distance_m']['p95'] is None else 100*s['step_distance_m']['p95'],
            'rotation_median_deg': s['adjacent_rotation_deg']['median'],
            'rotation_p95_deg': s['adjacent_rotation_deg']['p95'],
            'gapped_pairs': s['source_id_gaps']['pair_count'],
            **{k+'_circular_span_deg': v['circular_span_deg'] for k,v in s['source_euler_ranges'].items()}}


def export_statistics(result, output_dir):
    """Write reports to a new/empty directory, preserving previous analyses."""
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Choose a new/empty report directory: {out}')
    out.mkdir(parents=True, exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result['summary'], indent=2, allow_nan=False)+'\n', encoding='utf-8')
    with open(out/'comparison.csv', 'w', newline='', encoding='utf-8') as f:
        row = comparison_row(result)
        w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerow(row)
    with open(out/'frames.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['frame_id','source_frame_id','x_m','y_m','world_z_m','cumulative_distance_m',
                    'rotation_from_first_deg','source_pitch_deg','source_yaw_deg','source_roll_deg'])
        for i, (sid, xyz, euler) in enumerate(zip(result['source_ids'],result['xyz'],result['euler_deg'])):
            w.writerow([i,int(sid),*xyz,result['cumulative_distance_m'][i],result['rotation_from_first_deg'][i],*euler])
    with open(out/'pairs.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['from_frame_id','to_frame_id','from_source_id','to_source_id','source_id_delta',
                    'distance_m','distance_cm','dx_m','dy_m','dz_m','rotation_deg'])
        for i, distance in enumerate(result['step_distance_m']):
            w.writerow([i,i+1,int(result['source_ids'][i]),int(result['source_ids'][i+1]),
                        int(result['source_id_delta'][i]),distance,100*distance,
                        *result['delta_xyz_m'][i],result['rotation_step_deg'][i]])
    return out
