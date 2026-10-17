"""Package original ERP images with explicit camera-to-world GT; CPU only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from geometry.poses import (CARLA_CAMERA_FROM_ERP, WORLD_FROM_CARLA,
                            source_to_c2w, quaternion_xyzw, validate_poses)

SOURCE_COLUMNS = ['x', 'y', 'z', 'pitch_deg', 'yaw_deg', 'roll_deg']
MATCH_TOLERANCE = 0.00051
REPORT_COLUMNS = ['source_image_name', 'source_frame_id', 'status', 'match_method',
                  'source_csv_row', 'source_csv_line', 'candidate_csv_rows', 'reason']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def collect_records(dataset, pose_csv=None, match_mode='auto'):
    """Match original IDs first, or search all six pose values; never use nearest pose."""
    if match_mode not in ('auto', 'index', 'pose'):
        raise ValueError('match_mode must be auto, index or pose')
    dataset = Path(dataset)
    if not dataset.is_dir():
        raise ValueError(f'Dataset directory does not exist: {dataset}')
    source = None
    if pose_csv is not None:
        source = np.loadtxt(pose_csv, delimiter=',', ndmin=2, encoding='utf-8-sig')
        if source.shape[1] != 6 or not np.isfinite(source).all():
            raise ValueError('CSV must have six finite columns, no header: x,y,z,pitch,yaw,roll')
    records, report, seen = [], [], set()
    tree = None
    for path in dataset.glob('equi_*.png'):
        fields = path.stem.split('_')
        if len(fields) != 8:
            raise ValueError(f'Unexpected filename: {path.name}')
        index = int(fields[1])
        label = np.array(fields[2:], dtype=np.float64)
        if index < 0 or index in seen or not np.isfinite(label).all():
            raise ValueError(f'Duplicate, negative index or invalid pose: {path.name}')
        seen.add(index)
        entry = dict.fromkeys(REPORT_COLUMNS, '')
        entry.update(source_image_name=path.name, source_frame_id=index, status='matched')
        csv_row, method = None, 'filename'
        if source is not None:
            index_matches = index < len(source) and np.allclose(
                label, source[index], atol=MATCH_TOLERANCE, rtol=0)
            if match_mode != 'pose' and index_matches:
                candidates, method = [index], 'index_verified'
            elif match_mode == 'index':
                candidates = []
            else:
                if tree is None:
                    tree = cKDTree(source)
                candidates = sorted(tree.query_ball_point(label, MATCH_TOLERANCE, p=np.inf))
                method = 'pose_search'
            entry['candidate_csv_rows'] = json.dumps(candidates)
            entry['match_method'] = method if method != 'filename' else 'index_check'
            if len(candidates) != 1:
                entry.update(status='unmatched' if not candidates else 'ambiguous',
                             reason='No matching CSV row' if not candidates else
                             'Multiple rows match rounded filename pose; identity is uncertain')
                report.append(entry)
                continue
            csv_row = candidates[0]
            pose = source[csv_row].copy()
            entry.update(source_csv_row=csv_row, source_csv_line=csv_row+1)
        else:
            pose = label
        entry['match_method'] = method
        with Image.open(path) as im:
            width, height = im.size
            im.verify()
        if width != 2*height or height < 2:
            raise ValueError(f'Expected 2:1 ERP: {path.name}')
        records.append((index, path, pose, width, height, csv_row, method))
        report.append(entry)
    if not seen:
        raise ValueError('No equi_*.png images found in dataset directory')
    # Do not silently attach multiple images to one CSV row after a search.
    owners = {}
    for entry in report:
        if entry['status'] == 'matched' and entry['source_csv_row'] != '':
            owners.setdefault(entry['source_csv_row'], []).append(entry)
    for entries in owners.values():
        if len(entries) > 1:
            for entry in entries:
                entry.update(status='row_conflict', reason='Multiple images claim the same CSV row')
    accepted = {e['source_frame_id'] for e in report if e['status'] == 'matched'}
    return (sorted([r for r in records if r[0] in accepted], key=lambda r: r[0]),
            sorted(report, key=lambda e: e['source_frame_id']))


def build_standard_gt(dataset='dataset', output='Standard_GT', pose_csv=None,
                      sequence_id='sequence000', map_name=None, carla_version=None,
                      match_mode='auto', on_unmatched='error'):
    """Return sequence directory. Existing sequence directories are never overwritten.

    Uses supplied MATLAB geometry profile; original capture labels are accepted.
    Full-precision CSV is preferred. No CSV means rounded filename poses.
    """
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', sequence_id):
        raise ValueError('sequence_id must be a simple folder name')
    if on_unmatched not in ('error', 'skip'):
        raise ValueError('on_unmatched must be error or skip')
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if pose_csv is None and (dataset/'positions_angles_camera1.csv').exists():
        pose_csv = dataset/'positions_angles_camera1.csv'
    if pose_csv is not None:
        pose_csv = Path(pose_csv).resolve()
    target = output/sequence_id
    if target.exists():
        raise FileExistsError(f'Output sequence already exists: {target}')
    records, report = collect_records(dataset, pose_csv, match_mode)
    rejected = [e for e in report if e['status'] != 'matched']
    if (rejected and on_unmatched == 'error') or not records:
        details = '\n'.join(f"{e['source_image_name']}: {e['reason']} "
                            f"(candidate rows: {e['candidate_csv_rows']})" for e in rejected)
        raise ValueError(f'{len(rejected)} images could not be matched safely.\n{details}\n'
                         'Use on_unmatched="skip" / --on-unmatched skip to export other matched images. '
                         'At least one matched image is required.')
    poses = np.stack([source_to_c2w(r[2]) for r in records])
    checks = validate_poses(poses)
    quats = quaternion_xyzw(poses[:, :3, :3])
    checks['max_quaternion_roundtrip_error'] = float(np.max(np.abs(
        Rotation.from_quat(quats).as_matrix()-poses[:, :3, :3])))
    output.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.gt-build-', dir=output))
    try:
        (staging/'images').mkdir()
        columns = ['sequence_id', 'frame_id', 'source_frame_id', 'image_name',
                   'source_image_name', 'width', 'height', 'tx_m', 'ty_m', 'tz_m',
                   'qx', 'qy', 'qz', 'qw']
        columns += [f'r{i}{j}' for i in range(3) for j in range(3)]
        columns += ['source_'+s for s in SOURCE_COLUMNS] + ['image_sha256', 'source_csv_row',
                                                         'source_csv_line', 'match_method']
        with open(staging/'poses.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for new_id, (source_id, path, pose, w, h, csv_row, method) in enumerate(records):
                name = f'{new_id:06d}.png'
                destination = staging/'images'/name
                before = sha256(path)
                shutil.copy2(path, destination)
                if sha256(destination) != before:
                    raise IOError(f'Image copy verification failed: {path.name}')
                values = [*poses[new_id, :3, 3], *quats[new_id],
                          *poses[new_id, :3, :3].ravel(), *pose]
                writer.writerow([sequence_id, new_id, source_id, 'images/'+name,
                                 path.name, w, h, *[format(float(x), '.17g') for x in values], before,
                                 '' if csv_row is None else csv_row,
                                 '' if csv_row is None else csv_row+1, method])
        with open(staging/'matching_report.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
            writer.writeheader()
            writer.writerows(report)
        np.save(staging/'poses_c2w.npy', poses, allow_pickle=False)
        metadata = {
            'schema_version': '1.0', 'geometry_profile': 'source_matlab_v1',
            'sequence_id': sequence_id, 'map_name': map_name, 'carla_version': carla_version,
            'frame_count': len(records), 'pose_convention': 'camera_to_world',
            'equation': 'p_world = R_wc @ p_camera + t_wc',
            'world': {'handedness': 'right', 'axes': ['CARLA X', '-CARLA Y', 'CARLA Z'], 'unit': 'metre'},
            'camera': {'handedness': 'right', 'axes': ['right', 'down', 'forward']},
            'quaternion_order': ['qx', 'qy', 'qz', 'qw'], 'quaternion_sign': 'canonical w>=0',
            'matrix_storage': 'float64 (N,4,4), CSV row order; column-vector transforms',
            'erp': {'sampling': 'endpoint_inclusive', 'pixel_indexing': 'zero_based',
                    'longitude': '2*pi*u/(W-1)-pi', 'latitude': 'pi/2-pi*v/(H-1)',
                    'ray': '[cos(lat)*sin(lon), -sin(lat), cos(lat)*cos(lon)]',
                    'horizontal_seam': 'left/right edges are duplicate longitude',
                    'image_operation': 'byte_identical_copy_no_resampling'},
            'conversion': {'world_from_carla': WORLD_FROM_CARLA.tolist(),
                           'carla_camera_from_erp': CARLA_CAMERA_FROM_ERP.tolist(),
                           'rotation': 'world_from_carla @ R_carla @ carla_camera_from_erp',
                           'translation': 'world_from_carla @ t_carla',
                           'source_stitch_rotation': 'normalized quaternion wxyz=(1,1,0,0), Rx(+90deg)',
                           'source_indexing': 'MATLAB row=Xp,column=Yp'},
            'pose_source': 'csv' if pose_csv else 'filename_rounded',
            'matching': {'mode': match_mode, 'on_unmatched': on_unmatched,
                         'tolerance_per_source_component': MATCH_TOLERANCE,
                         'input_count': len(report), 'exported_count': len(records),
                         'skipped_count': len(rejected), 'report': 'matching_report.csv',
                         'row_indexing': 'source_csv_row zero-based; source_csv_line one-based',
                         'index_assumption': 'auto/index trust original filename IDs if all six values agree'},
            'source_csv': None if pose_csv is None else {'name': pose_csv.name, 'sha256': sha256(pose_csv)},
            'precision_note': 'Full supplied CSV precision' if pose_csv else 'Limited by filename rounding; lost precision is not restored',
            'validation': {'conversion_checks': checks, 'image_hashes_verified': True,
                           'capture_alignment': 'accepted_from_input_not_independently_verified',
                           'image_pose_reprojection': 'not_performed_no_scene_landmarks'},
            'timestamps': None,
        }
        (staging/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
        # Rename publishes a complete sequence; cleanup below only owns staging.
        if target.exists():
            raise FileExistsError(target)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default='dataset')
    p.add_argument('--output', default='Standard_GT')
    p.add_argument('--pose-csv')
    p.add_argument('--sequence-id', default='sequence000')
    p.add_argument('--map-name')
    p.add_argument('--carla-version')
    p.add_argument('--match-mode', choices=['auto', 'index', 'pose'], default='auto',
                   help='auto: verify index then search; pose: search without trusting index')
    p.add_argument('--on-unmatched', choices=['error', 'skip'], default='error',
                   help='skip unmatched, ambiguous and conflicting rows, recording the report')
    a = p.parse_args()
    try:
        result = build_standard_gt(**vars(a))
    except (ValueError, OSError) as e:
        p.exit(1, f'Export failed: {e}\n')
    meta = json.loads((result/'metadata.json').read_text())
    print(f'Exported {meta["frame_count"]} frames to {result}')
    print('Pose source:', meta['pose_source'])
    print('Skipped images:', meta['matching']['skipped_count'])
    print('Matching report:', result/'matching_report.csv')
    print('Conversion checks:', meta['validation']['conversion_checks'])
    print('Images copied without pixel changes. Capture alignment accepted from input.')


if __name__ == '__main__':
    main()
