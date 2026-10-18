import csv,json,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
from comparison.projection import FACE_ROTATIONS,face_rays,erp_grid,extract_faces,merge_faces,z_to_radial,z_to_radial_intrinsics
from comparison.prepare import prepare_job

class ComparisonTests(unittest.TestCase):
    def test_rotations_and_sphere(self):
        for r in FACE_ROTATIONS:
            np.testing.assert_allclose(r.T@r,np.eye(3));self.assertAlmostEqual(np.linalg.det(r),1)
        # A radius-7 sphere must stay constant after z->radial conversion and fusion.
        z=7/np.linalg.norm(face_rays(70),axis=-1)
        d=merge_faces([z_to_radial(z)]*6,64,128)
        self.assertTrue(np.isfinite(d).all());np.testing.assert_allclose(d,7,atol=1e-5)
    def test_predicted_intrinsics_depth_conversion(self):
        size=71;focal=(size-1)/2/np.tan(np.deg2rad(100)/2)
        K=np.array([[focal,0,35],[0,focal,35],[0,0,1]])
        depth=7/np.linalg.norm(face_rays(size),axis=-1)
        np.testing.assert_allclose(z_to_radial_intrinsics(depth,K),7,atol=1e-10)
    def test_direction_texture(self):
        rgb=np.uint8((erp_grid(180,360)+1)*127.5)
        faces=extract_faces(rgb,71)
        for face,r in zip(faces,FACE_ROTATIONS):
            np.testing.assert_allclose(face[35,35],(r[:,2]+1)*127.5,atol=3)
    def test_invalid_depth_not_filled(self):
        out=merge_faces([np.full((28,28),np.nan)]*6,32,64)
        self.assertTrue(np.isnan(out).all())
    def test_prepare_no_poses_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);seq=root/'seq';seq.mkdir()
            Image.fromarray(np.zeros((32,64,3),np.uint8)).save(seq/'test.png')
            (seq/'poses.csv').write_text('frame_id,image_name,source_frame_id\n0,test.png,10\n')
            out=prepare_job(seq,root/'job',[0],height=32,face_size=28)
            mf=json.loads((out/'manifest.json').read_text())
            self.assertFalse(mf['gt_used_as_model_input']);self.assertEqual(len(mf['frames'][0]['faces']),6)
            self.assertNotIn('pose',mf['frames'][0])
            with self.assertRaises(FileExistsError):prepare_job(seq,out,[0])
if __name__=='__main__':unittest.main()
