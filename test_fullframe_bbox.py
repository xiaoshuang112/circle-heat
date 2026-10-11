import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from prepare_fullframe_labelme import read_pairs
from train import SlotDataset, crop_bbox, letterbox, restore_centers


class FullFrameTests(unittest.TestCase):
    def test_jitter_pixels_and_coordinate_roundtrip(self):
        image = np.arange(121 * 203 * 3).reshape(121, 203, 3).astype(np.uint8)
        origins = set()
        for seed in range(100):
            random.seed(seed)
            crop, x, y, offset = crop_bbox(image, 100.25, 59.75, [30.2, 20.4, 170.8, 100.3], True)
            left, top = offset
            origins.add(offset)
            self.assertLessEqual(abs(left - 30), 4)
            self.assertLessEqual(abs(top - 20), 2)
            np.testing.assert_array_equal(crop, image[top:top + 81, left:left + 141])
            _, tx, ty, meta = letterbox(crop, x, y)
            meta[2:] -= np.asarray(offset) * meta[:2]
            original = (np.array([tx, ty]) - meta[2:] + .5) / meta[:2] - .5
            np.testing.assert_allclose(original, [100.25, 59.75], atol=2e-5, rtol=0)
        self.assertGreater(len(origins), 10)
        for bbox in ([0, 0, float('nan'), 10], [-1, 0, 100, 100], [5, 5, 5, 10]):
            with self.assertRaises(ValueError):
                crop_bbox(image, 50, 50, bbox)
        with self.assertRaises(ValueError):
            crop_bbox(image, 0, 0, [30, 20, 170, 100])

    def test_pair_validation_and_dataset_alignment(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cv2.imwrite(str(root / 'sample.png'), np.full((121, 203, 3), 100, np.uint8))
            shapes = [dict(label='bbox', shape_type='rectangle', group_id=1, points=[[30, 20], [170, 100]]),
                      dict(label='occupied', shape_type='circle', group_id=1, points=[[100.25, 59.75], [150.25, 59.75]])]
            annotation = root / 'sample.json'
            data = dict(imagePath='sample.png', imageWidth=203, imageHeight=121, shapes=shapes)
            annotation.write_text(json.dumps(data))
            _, _, pairs = read_pairs(annotation)
            self.assertEqual(pairs[0]['occupancy'], 'occupied')
            labels = root / 'labels.jsonl'
            labels.write_text(json.dumps(dict(image='sample.png', split='val', label_state='localizable', **pairs[0])) + '\n')
            dataset = SlotDataset(labels, 'val')
            baseline = dataset[0]
            with patch('train.random.random', side_effect=AssertionError('validation jittered')):
                for key, value in dataset[0].items():
                    self.assertTrue(torch.equal(value, baseline[key]))
            dataset = SlotDataset(labels, 'val', augment=True)
            for seed in range(30):
                random.seed(seed)
                sample = dataset[0]
                recovered = restore_centers(sample['center'][None], sample['meta'][None], sample['warp_inverse'][None])[0]
                torch.testing.assert_close(recovered, torch.tensor([100.25, 59.75]), atol=1e-4, rtol=0)
                self.assertTrue(torch.isfinite(sample['heatmap']).all())
            data['shapes'] = shapes[:1]
            annotation.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                read_pairs(annotation)
            data['shapes'] = shapes + [shapes[1]]
            annotation.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                read_pairs(annotation)


if __name__ == '__main__':
    unittest.main()
