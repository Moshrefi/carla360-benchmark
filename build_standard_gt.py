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
from geometry.poses import (CARLA_CAMERA_FROM_ERP, WORLD_FROM_CARLA,
                            source_to_c2w, quaternion_xyzw, validate_poses)

SOURCE_COLUMNS = ['x', 'y', 'z', 'pitch_deg', 'yaw_deg', 'roll_deg']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def collect_records(dataset, pose_csv=None):
    dataset = Path(dataset)
    if not dataset.is_dir():
        raise ValueError(f'Dataset directory does not exist: {dataset}')
    source = None
    if pose_csv is not None:
        source = np.loadtxt(pose_csv, delimiter=',', ndmin=2, encoding='utf-8-sig')
        if source.shape[1] != 6 or not np.isfinite(source).all():
            raise ValueError('CSV must have six finite columns, no header: x,y,z,pitch,yaw,roll')
    records, seen = [], set()
    for path in dataset.glob('equi_*.png'):
        fields = path.stem.split('_')
        if len(fields) != 8:
            raise ValueError(f'Unexpected filename: {path.name}')
        index = int(fields[1])
        label = np.array(fields[2:], dtype=np.float64)
        if index < 0 or index in seen or not np.isfinite(label).all():
            raise ValueError(f'Duplicate, negative index or invalid pose: {path.name}')
        seen.add(index)
        if source is not None:
            if index >= len(source):
                raise ValueError(f'No CSV row for frame {index}')
            pose = source[index].copy()
            if not np.allclose(label, pose, atol=0.00051, rtol=0):
                raise ValueError(f'Filename does not match zero-based CSV row {index}: {path.name}')
        else:
            pose = label
        with Image.open(path) as im:
            width, height = im.size
            im.verify()
        if width != 2*height or height < 2:
            raise ValueError(f'Expected 2:1 ERP: {path.name}')
        records.append((index, path, pose, width, height))
    if not records:
        raise ValueError('No equi_*.png images found in dataset directory')
    return sorted(records, key=lambda item: item[0])


def build_standard_gt(dataset='dataset', output='Standard_GT', pose_csv=None,
                      sequence_id='sequence000', map_name=None, carla_version=None):
    """Return sequence directory. Existing sequence directories are never overwritten.

    Uses supplied MATLAB geometry profile; original capture labels are accepted.
    Full-precision CSV is preferred. No CSV means rounded filename poses.
    """
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', sequence_id):
        raise ValueError('sequence_id must be a simple folder name')
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if pose_csv is None and (dataset/'positions_angles_camera1.csv').exists():
        pose_csv = dataset/'positions_angles_camera1.csv'
    if pose_csv is not None:
        pose_csv = Path(pose_csv).resolve()
    target = output/sequence_id
    if target.exists():
        raise FileExistsError(f'Output sequence already exists: {target}')
    records = collect_records(dataset, pose_csv)
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
        columns += ['source_'+s for s in SOURCE_COLUMNS] + ['image_sha256']
        with open(staging/'poses.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for new_id, (source_id, path, pose, w, h) in enumerate(records):
                name = f'{new_id:06d}.png'
                destination = staging/'images'/name
                before = sha256(path)
                shutil.copy2(path, destination)
                if sha256(destination) != before:
                    raise IOError(f'Image copy verification failed: {path.name}')
                values = [*poses[new_id, :3, 3], *quats[new_id],
                          *poses[new_id, :3, :3].ravel(), *pose]
                writer.writerow([sequence_id, new_id, source_id, 'images/'+name,
                                 path.name, w, h, *[format(float(x), '.17g') for x in values], before])
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
    a = p.parse_args()
    try:
        result = build_standard_gt(**vars(a))
    except (ValueError, OSError) as e:
        p.exit(1, f'Export failed: {e}\n')
    meta = json.loads((result/'metadata.json').read_text())
    print(f'Exported {meta["frame_count"]} frames to {result}')
    print('Pose source:', meta['pose_source'])
    print('Conversion checks:', meta['validation']['conversion_checks'])
    print('Images copied without pixel changes. Capture alignment accepted from input.')


if __name__ == '__main__':
    main()
