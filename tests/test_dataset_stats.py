import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from analysis.dataset_stats import pose_metrics, circular_span_deg, export_statistics


class StatisticsTests(unittest.TestCase):
    def test_known_translation_rotation_and_gaps(self):
        poses=np.tile(np.eye(4),(3,1,1))
        poses[:,:3,3]=[[0,0,2],[3,4,2],[3,4,2]]
        poses[:,:3,:3]=Rotation.from_euler('xyz',[[0,0,0],[0,0,90],[0,0,180]],degrees=True).as_matrix()
        r=pose_metrics(poses,[4,5,8],np.zeros((3,3)))
        np.testing.assert_allclose(r['step_distance_m'],[5,0])
        np.testing.assert_allclose(r['rotation_step_deg'],[90,90])
        s=r['summary'];self.assertEqual(s['retained_path_length_m'],5)
        self.assertEqual(s['source_id_gaps']['missing_ids_between_retained_frames'],2)
        self.assertEqual(s['low_translation_high_rotation_pair_count'],1)
        self.assertEqual(s['consecutive_source_id_step_distance_m']['mean'],5)

    def test_wraparound_is_two_degrees(self):
        poses=np.tile(np.eye(4),(2,1,1))
        poses[:,:3,:3]=Rotation.from_euler('xyz',[[0,0,179],[0,0,-179]],degrees=True).as_matrix()
        result=pose_metrics(poses,[0,1],[[0,179,0],[0,-179,0]])
        np.testing.assert_allclose(result['rotation_step_deg'],[2])
        self.assertEqual(circular_span_deg([179,-179]),2)
        self.assertEqual(result['summary']['source_euler_ranges']['yaw']['raw_span_deg'],358)

    def test_single_frame_and_report(self):
        result=pose_metrics(np.eye(4)[None],[9],[[0,0,0]])
        self.assertEqual(result['summary']['pair_count'],0)
        self.assertIsNone(result['summary']['step_distance_m']['mean'])
        with tempfile.TemporaryDirectory() as tmp:
            out=export_statistics(result,Path(tmp)/'report')
            self.assertEqual(json.loads((out/'summary.json').read_text())['frame_count'],1)
            self.assertEqual(len((out/'pairs.csv').read_text().splitlines()),1)
            with self.assertRaises(FileExistsError):export_statistics(result,out)

    def test_reject_unsorted_ids_and_negative_threshold(self):
        poses=np.tile(np.eye(4),(2,1,1))
        with self.assertRaises(ValueError):pose_metrics(poses,[2,1],np.zeros((2,3)))
        with self.assertRaises(ValueError):pose_metrics(poses,[1,2],np.zeros((2,3)),-1)
