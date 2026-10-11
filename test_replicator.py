import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from generate_replicator import frame_labels, project_points, read_intrinsics, rebase_records
from prepare_fullframe_labelme import read_pairs
from train import SlotDataset, load_records


class ReplicatorTests(unittest.TestCase):
    def setUp(self):
        self.view = np.eye(4)
        self.view[3, 2] = -.7
        self.projection = np.zeros((4, 4))
        self.projection[0, 0] = self.projection[1, 1] = 70 / 36
        self.projection[2, 2] = -(10 + .01) / (10 - .01)
        self.projection[2, 3] = -1
        self.projection[3, 2] = -2 * 10 * .01 / (10 - .01)

    def test_projection_axes_and_invalid_points(self):
        pixels = project_points([[0, 0, 0], [.1, 0, 0], [0, .1, 0]],
                                self.view, self.projection, 768, 768)
        np.testing.assert_allclose(pixels[0], [383.5, 383.5])
        self.assertGreater(pixels[1, 0], pixels[0, 0])
        self.assertLess(pixels[2, 1], pixels[0, 1])
        for points in ([[0, 0, 1]], [[5, 0, 0]], [[float('nan'), 0, 0]], [[0, 0, -.001 + .7]]):
            with self.assertRaises(ValueError):
                project_points(points, self.view, self.projection, 768, 768)

    def test_calibration_scaling_and_distorted_projection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'camera.yaml'
            k = np.array([[600., 0, 301.], [0, 620., 247.], [0, 0, 1.]])
            d = np.array([.1, 0, 0, 0, 0.])
            storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
            for key, value in dict(image_width=640, image_height=480,
                                   camera_matrix=k, distortion_coefficients=d).items():
                storage.write(key, value)
            storage.release()
            scaled, distortion, height = read_intrinsics(path, 320, None)
            self.assertEqual(height, 240)
            np.testing.assert_allclose(scaled[:2], k[:2] * .5)
            np.testing.assert_array_equal(distortion, d)
            pixels = project_points([[0, 0, 0], [.14, 0, 0], [0, .14, 0]],
                                    self.view, None, 320, height, (scaled, distortion))
            np.testing.assert_allclose(pixels, [[150.5, 123.5],
                                              [150.5 + 300 * .2 * 1.004, 123.5],
                                              [150.5, 123.5 - 310 * .2 * 1.004]])
            with self.assertRaises(ValueError):
                project_points([[0, 0, 1]], self.view, None, 320, height, (scaled, distortion))

    def test_labelme_and_training_roundtrip_preserve_validation(self):
        slots = [(0., 0., (0., 0., .002), False), (.1, 0., (.103, -.002, .024), True)]
        k = np.array([[700., 0, 382.], [0, 705., 386.], [0, 0, 1.]])
        records, annotation = frame_labels('sample.png', slots, self.view, self.projection,
                                          768, 768, (k, np.array([-.05, .14, -.0004, .00005, -.07])))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images = root / 'labelme'
            images.mkdir()
            cv2.imwrite(str(images / 'sample.png'), np.full((768, 768, 3), 100, np.uint8))
            labelme = images / 'sample.json'
            labelme.write_text(json.dumps(annotation))
            _, _, pairs = read_pairs(labelme)
            self.assertEqual([p['occupancy'] for p in pairs], ['empty', 'occupied'])
            for shape in annotation['shapes'][1::2]:
                self.assertAlmostEqual(np.linalg.norm(np.subtract(*shape['points'])), 50.)
            baseline = root / 'baseline.jsonl'
            real = dict(records[0], split='val', split_group='real-session', synthetic=False)
            baseline.write_text(json.dumps(real) + '\n')
            old_bytes = baseline.read_bytes()
            rebased = rebase_records(baseline)
            self.assertEqual(rebased[0], dict(real, image=str(images / 'sample.png')))
            combined = root / 'labels.jsonl'
            combined.write_text(''.join(json.dumps(r) + '\n' for r in rebased + records))
            self.assertEqual(baseline.read_bytes(), old_bytes)
            self.assertEqual(len(load_records(combined)), 3)
            dataset = SlotDataset(combined, 'train')
            self.assertEqual(len(dataset), 2)
            self.assertEqual(len(SlotDataset(combined, 'val')), 1)
            for index, record in enumerate(records):
                sample = dataset[index]
                meta = sample['meta']
                recovered = (sample['center'] - meta[2:] + .5) / meta[:2] - .5
                torch.testing.assert_close(recovered, torch.tensor([record['center_x'], record['center_y']]),
                                           atol=1e-4, rtol=0)
            real['synthetic'] = True
            baseline.write_text(json.dumps(real) + '\n')
            with self.assertRaises(ValueError):
                rebase_records(baseline)


if __name__ == '__main__':
    unittest.main()
