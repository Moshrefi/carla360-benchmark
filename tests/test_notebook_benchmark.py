"""CPU tests only; synthetic predictions do not validate model inference."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from comparison import bench_ui as b


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / 'input'
        self.data.mkdir()
        for n in (1, 2, 10, 11, 12):
            Image.new('RGB', (16, 8), (n, 20, 30)).save(self.data / f'equi_{n}.png')
        self.paths = b.select_images(self.data)
        self.plan = b.make_plan(self.paths, self.root / 'runs/a', {}, ['vggt', 'argus'], 2)

    def tearDown(self):
        plt.close('all')
        self.tmp.cleanup()

    def test_selection_chunking_and_manifest_guard(self):
        self.assertEqual([p.stem for p in self.paths], ['equi_1', 'equi_2', 'equi_10', 'equi_11', 'equi_12'])
        self.assertEqual(len(b.select_images(self.data, start=1, stride=2, limit=2)), 2)
        chunks = b.prepare(self.plan)
        self.assertEqual([len(records) for _, records, _ in chunks], [2, 2, 1])
        for chunk, records, staged in chunks:
            for i, record in enumerate(records):
                self.assertEqual(b.digest(staged / f'{i:06d}.png'), record['sha256'])
        b.prepare(self.plan)
        with self.assertRaises(ValueError):
            b.prepare({**self.plan, 'chunk_size': 3})
        Image.new('RGB', (16, 8)).save(self.paths[0])
        with self.assertRaises(ValueError):
            b.prepare(self.plan)

    def test_output_integrity_identity_and_plots(self):
        chunk, records, _ = b.prepare(self.plan)[0]
        out = chunk / 'vggt'; out.mkdir()
        np.savez_compressed(out / 'predictions.npz', depth=np.ones((2, 8, 16)),
            extrinsics=np.tile(np.eye(4)[:3], (2, 1, 1)), world_points=np.ones((2, 8, 16, 3)))
        meta = {'input_images': records, 'request_hash': 'test', 'output_sha256': b.digest(out / 'predictions.npz')}
        (out / 'metadata.json').write_text(json.dumps(meta))
        self.assertEqual(b.load_result(out, 'test')[0]['depth'].shape, (2, 8, 16))
        with self.assertRaises(ValueError): b.load_result(out, 'different')
        fig = b.depth_plot(self.plan); fig.savefig(self.root / 'plot.png')
        figures, rows = b.geometry_plots(self.plan, max_points=5)
        self.assertEqual(len(figures), 2)
        self.assertTrue(rows)
        (out / 'predictions.npz').write_bytes(b'corrupt')
        with self.assertRaises(ValueError): b.load_result(out)

    def test_argus_never_launches_process(self):
        plan = {**self.plan, 'selected': ['argus']}
        with patch.object(b.subprocess, 'Popen', side_effect=AssertionError('Must not launch')):
            statuses = b.execute(plan, '/not/used')
        self.assertIn('pending', statuses[0]['status'])


if __name__ == '__main__':
    unittest.main()
