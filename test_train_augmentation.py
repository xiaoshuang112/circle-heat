import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from train import IMAGENET_MEAN, IMAGENET_STD, SlotDataset, perspective_augment, restore_centers


class AugmentationTests(unittest.TestCase):
    def test_perspective_pixels_and_inverse(self):
        yy, xx = np.mgrid[:320, :320]
        image = np.repeat((255 * np.exp(-((xx-101.25)**2+(yy-137.75)**2)/18))[..., None], 3, axis=2).astype(np.uint8)
        for seed in range(30):
            random.seed(seed)
            warped, x, y, inverse = perspective_augment(image, 101.25, 137.75)
            recovered = restore_centers(torch.tensor([[x,y]]), torch.tensor([[1.,1.,0.,0.]]), torch.from_numpy(inverse)[None])
            torch.testing.assert_close(recovered, torch.tensor([[101.25,137.75]]), atol=1e-4, rtol=0)
            # Ignore the synthetic pad color along the warped image boundary.
            signal = warped[110:160, 75:130, 0].astype(float)
            sy, sx = np.mgrid[110:160, 75:130]
            np.testing.assert_allclose([(sx*signal).sum()/signal.sum(), (sy*signal).sum()/signal.sum()], [x,y], atol=.2, rtol=0)
        with patch('train.random.uniform', return_value=-4.8):
            original, x, y, inverse = perspective_augment(image, 0., 0.)
            self.assertIs(original, image)
            self.assertEqual((x,y), (0.,0.))
            np.testing.assert_array_equal(inverse, np.eye(3))

    def test_photometric_blur_and_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = np.full((320,320,3), 80, np.uint8)
            image[:,160:] = 180
            cv2.imwrite(str(root/'image.png'), image)
            record = dict(image='image.png', split='val', label_state='localizable', occupancy='occupied', center_x=100., center_y=140.)
            labels = root/'labels.jsonl'
            labels.write_text(json.dumps(record)+'\n')
            plain = SlotDataset(labels,'val')[0]
            with patch('train.random.random', side_effect=AssertionError('validation augmented')):
                unchanged = SlotDataset(labels,'val')[0]
            for key in plain:
                self.assertTrue(torch.equal(plain[key],unchanged[key]))
            dataset = SlotDataset(labels,'val',augment=True)
            with patch('train.random.randint', return_value=0), patch('train.random.random', return_value=1.), patch('train.random.uniform', side_effect=[1.1,12.]):
                sample = dataset[0]
            decoded = (sample['image'].numpy().transpose(1,2,0)*IMAGENET_STD+IMAGENET_MEAN)*255
            np.testing.assert_allclose(decoded[140,100], 100, atol=1e-4)
            np.testing.assert_allclose(decoded[140,200], 210, atol=1e-4)
            with patch('train.random.randint',return_value=0), patch('train.random.random',side_effect=[1.,0.]), patch('train.random.uniform',side_effect=[1.,0.,.8]), patch('train.cv2.GaussianBlur',wraps=cv2.GaussianBlur) as blur:
                blurred = dataset[0]
            blur.assert_called_once()
            self.assertFalse(torch.equal(blurred['image'],plain['image']))
            self.assertTrue(torch.equal(blurred['center'],plain['center']))
            for seed in range(100):
                random.seed(seed)
                sample = dataset[0]
                restored = restore_centers(sample['center'][None],sample['meta'][None],sample['warp_inverse'][None])[0]
                torch.testing.assert_close(restored,torch.tensor([100.,140.]),atol=1e-4,rtol=0)
                self.assertTrue(torch.isfinite(sample['heatmap']).all())
                self.assertAlmostEqual(float(sample['heatmap'].sum()),1.,places=5)
                self.assertEqual(float(sample['occupancy']),1.)


if __name__ == '__main__':
    unittest.main()
