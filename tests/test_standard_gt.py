import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from build_standard_gt import build_standard_gt, sha256
from datasets.standard_gt import StandardGTDataset
from geometry.poses import (carla_rotation, erp_rays, relative_c2w,
                            source_to_c2w, source_transform_from_c2w, validate_poses)


class GeometryTests(unittest.TestCase):
    def test_cardinal_rays_and_seam(self):
        rays = erp_rays([2, 3, 2, 2, 0, 4], [1, 1, 0, 2, 1, 1], 5, 3)
        np.testing.assert_allclose(rays, [[0,0,1],[1,0,0],[0,-1,0],
                                         [0,1,0],[0,0,-1],[0,0,-1]], atol=1e-15)

    def test_known_carla_rotations(self):
        np.testing.assert_allclose(carla_rotation(0,90,0) @ [1,0,0], [0,1,0], atol=1e-15)
        np.testing.assert_allclose(carla_rotation(90,0,0) @ [1,0,0], [0,0,1], atol=1e-15)
        np.testing.assert_allclose(carla_rotation(0,0,90) @ [0,1,0], [0,0,-1], atol=1e-15)
        np.testing.assert_allclose(source_to_c2w([1,2,3,0,0,0]),
                                  [[-1,0,0,1],[0,-1,0,-2],[0,0,1,3],[0,0,0,1]], atol=1e-15)

    def test_independent_source_ray_chain(self):
        # Construct rays with the original MATLAB equations, including both
        # hemispheres, then simulate row/column indexing and optical axes.
        rng = np.random.default_rng(12)
        u, v = rng.uniform(0,1439,2000), rng.uniform(0,719,2000)
        phi, theta = 2*np.pi*u/1439, np.pi*v/719
        x = np.sin(theta)*np.cos(phi)
        y = -np.cos(theta)             # Rx(+90) applied directly
        z = np.sin(theta)*np.sin(phi)
        is_front = z > 0
        xm = x/(1.92+np.abs(z))
        ym = np.where(is_front, y, -y)/(1.92+np.abs(z))
        # MATLAB samples row=Xp, col=Yp; generator rays assign X to col.
        xx, yy = ym, xm
        radius2 = xx*xx+yy*yy
        lam = (1.92+np.sqrt(1+(1-1.92**2)*radius2))/(1+radius2)
        optical = np.stack([lam*xx, lam*yy, lam-1.92], axis=-1)
        native = optical[:, [2,0,1]] * [1,1,-1]
        native[~is_front] *= [-1,-1,1]  # actual local CARLA back rotation
        for pitch,yaw,roll in [(0,0,0),(90,0,75.732),(74.28,26.899,89.962),(-20,150,30)]:
            # Independently compose elementary rotations for CARLA convention.
            rc = Rotation.from_euler('xyz',[-roll,-pitch,yaw],degrees=True).as_matrix()
            expected = (native @ rc.T) * [1,-1,1]
            exported = erp_rays(u,v,1440,720) @ source_to_c2w([0,0,0,pitch,yaw,roll])[:3,:3].T
            np.testing.assert_allclose(exported, expected, atol=1e-12)

    def test_roundtrip_including_gimbal_lock(self):
        rng = np.random.default_rng(7)
        src = rng.uniform(-180,180,(100,6))
        src[:2,3] = [90,-90]
        poses = np.stack([source_to_c2w(s) for s in src])
        validate_poses(poses)
        for s,t in zip(src,poses):
            original = source_transform_from_c2w(t)
            np.testing.assert_allclose(original[:3,3],s[:3],atol=1e-12)
            np.testing.assert_allclose(original[:3,:3],
                Rotation.from_euler('xyz',[-s[5],-s[3],s[4]],degrees=True).as_matrix(),atol=1e-12)

    def test_relative_transform_maps_points(self):
        a=source_to_c2w([2,-3,5,10,20,30]);b=source_to_c2w([6,4,2,-40,80,15])
        point=np.array([3,4,5,1])
        np.testing.assert_allclose(a @ relative_c2w(a,b) @ point,b @ point,atol=1e-12)


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.data=self.root/'dataset';self.data.mkdir()
        self.csv=self.root/'poses.csv'
        self.source=np.zeros((11,6));self.source[:,0]=1.123456789012345
        self.source[:,3]=90
        np.savetxt(self.csv,self.source,delimiter=',',fmt='%.17g')
        for i in [10,2,3]:
            Image.new('RGB',(8,4),(i,20,40)).save(self.data/f'equi_{i}_1.123_0.000_0.000_90.000_0.000_0.000.png')

    def export(self, **kw):
        return build_standard_gt(self.data,self.root/'Standard_GT',**kw)

    def test_precision_order_copy_and_csv_matrix_agreement(self):
        out=self.export(pose_csv=self.csv)
        with open(out/'poses.csv',newline='') as f:rows=list(csv.DictReader(f))
        self.assertEqual([int(r['source_frame_id']) for r in rows],[2,3,10])
        poses=np.load(out/'poses_c2w.npy',allow_pickle=False)
        for i,row in enumerate(rows):
            self.assertEqual(float(row['tx_m']),self.source[0,0])
            self.assertEqual(sha256(out/row['image_name']),sha256(self.data/row['source_image_name']))
            np.testing.assert_allclose(Rotation.from_quat([float(row[k]) for k in ['qx','qy','qz','qw']]).as_matrix(),poses[i,:3,:3],atol=1e-15)
            np.testing.assert_allclose([float(row[f'r{a}{b}']) for a in range(3) for b in range(3)],poses[i,:3,:3].ravel())
        self.assertEqual(json.loads((out/'metadata.json').read_text())['pose_source'],'csv')

    def test_filename_fallback_explicitly_recorded(self):
        out=self.export()
        self.assertEqual(json.loads((out/'metadata.json').read_text())['pose_source'],'filename_rounded')
        self.assertEqual(np.load(out/'poses_c2w.npy')[0,0,3],1.123)

    def test_standard_loader(self):
        out=self.export(pose_csv=self.csv)
        dataset=StandardGTDataset(out)
        self.assertEqual(len(dataset),3)
        self.assertEqual(dataset[0]['rgb'].shape,(4,8,3))
        self.assertEqual(dataset.manifest()[0]['source_frame_id'],2)
        np.testing.assert_allclose(dataset[0]['pose_c2w'],source_to_c2w(self.source[2]))

    def test_existing_output_refused(self):
        out=self.export(pose_csv=self.csv);before=sha256(out/'poses.csv')
        with self.assertRaises(FileExistsError):self.export(pose_csv=self.csv)
        self.assertEqual(sha256(out/'poses.csv'),before)

    def test_mismatch_does_not_publish(self):
        self.source[2,0]=100;np.savetxt(self.csv,self.source,delimiter=',')
        with self.assertRaises(ValueError):self.export(pose_csv=self.csv)
        self.assertFalse((self.root/'Standard_GT'/'sequence000').exists())

    def test_duplicate_index_refused(self):
        Image.new('RGB',(8,4)).save(self.data/'equi_2_1.123_0_0_90_0_0.png')
        with self.assertRaises(ValueError):self.export()

    def test_nonfinite_pose_refused(self):
        self.source[2,0]=np.nan;np.savetxt(self.csv,self.source,delimiter=',')
        with self.assertRaises(ValueError):self.export(pose_csv=self.csv)


if __name__ == '__main__':
    unittest.main()
