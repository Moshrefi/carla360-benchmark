import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from diagnostics.rotation_check import estimate_rotation, angular_error, choose_groups, save_report

class RotationDiagnosticTests(unittest.TestCase):
    def test_rotation_with_outliers_and_holdout(self):
        rng=np.random.default_rng(22)
        a=rng.normal(size=(300,3)); a/=np.linalg.norm(a,axis=1)[:,None]
        expected=Rotation.from_euler('xyz',[12,-8,23],degrees=True).as_matrix()
        b=a@expected.T
        b[:70]=rng.normal(size=(70,3)); b/=np.linalg.norm(b,axis=1)[:,None]
        r,q,_=estimate_rotation(a,b,iterations=500)
        self.assertTrue(q['quality_pass'])
        self.assertLess(angular_error(r,expected),1e-8)
        self.assertEqual(q['heldout_matches'],90)
    def test_groups_exclude_calibration_and_gaps(self):
        ids=np.delete(np.arange(100),[33,34])
        ds=SimpleNamespace(rows=[{'source_frame_id':int(i)} for i in ids])
        src=np.zeros((100,6))
        groups=choose_groups(ds,src,ids)
        self.assertEqual({g['region'] for g in groups},{'beginning','middle','end'})
        for g in groups:
            selected=ids[g['frames']]
            self.assertTrue((selected>5).all())
            self.assertTrue((np.diff(selected)==1).all())
        src[:,:3]=np.arange(100)[:,None]
        self.assertEqual(choose_groups(ds,src,ids),[])
    def test_reports_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'report'
            save_report({'results':[{'status':'usable','error':.1}]},p)
            with self.assertRaises(FileExistsError): save_report({'results':[]},p)
    def test_few_features_inconclusive(self):
        with self.assertRaises(ValueError): estimate_rotation(np.ones((3,3)),np.ones((3,3)))
