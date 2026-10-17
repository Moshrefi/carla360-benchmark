"""Read a sequence created by build_standard_gt.py."""
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
from geometry.poses import validate_poses


class StandardGTDataset:
    def __init__(self, sequence_dir):
        self.root = Path(sequence_dir).resolve()
        self.metadata = json.loads((self.root/'metadata.json').read_text(encoding='utf-8'))
        if (self.metadata.get('schema_version') != '1.0' or
                self.metadata.get('pose_convention') != 'camera_to_world'):
            raise ValueError('Unsupported Standard_GT schema or pose convention')
        with open(self.root/'poses.csv', newline='', encoding='utf-8') as f:
            self.rows = list(csv.DictReader(f))
        self.poses = np.load(self.root/'poses_c2w.npy', allow_pickle=False)
        validate_poses(self.poses)
        if len(self.rows) != len(self.poses) or len(self.rows) != self.metadata['frame_count']:
            raise ValueError('CSV / matrix / metadata frame counts differ')
        self.paths = []
        for i, row in enumerate(self.rows):
            path = (self.root/row['image_name']).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise ValueError('Invalid image path in poses.csv')
            if int(row['frame_id']) != i:
                raise ValueError('Unexpected row order in poses.csv')
            self.paths.append(path)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.paths[index]) as im:
            rgb = np.array(im.convert('RGB'))
        return {'frame_id': int(row['frame_id']), 'path': str(self.paths[index]), 'rgb': rgb,
                'position_carla': np.array([float(row['source_'+k]) for k in ['x','y','z']]),
                'pitch_yaw_roll_deg': np.array([float(row['source_'+k+'_deg']) for k in ['pitch','yaw','roll']]),
                'pose_c2w': self.poses[index].copy()}

    def manifest(self):
        return [{'frame_id': i, 'source_frame_id': int(row['source_frame_id']),
                 'path': str(self.paths[i]), 'pose_c2w': self.poses[i].tolist(),
                 'geometry_profile': self.metadata['geometry_profile']}
                for i, row in enumerate(self.rows)]
