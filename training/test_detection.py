import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

import detection
from utils import processing


class DetectionTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(42).integers(0, 256, (32, 64, 3), dtype=np.uint8)

    def test_no_detection_returns_no_card(self):
        model = Mock(return_value=[SimpleNamespace(obb=[])])
        self.assertEqual(detection.detect_card(model, self.image, [], Mock()), (None, None))

    def test_inference_errors_are_not_hidden(self):
        model = Mock(side_effect=RuntimeError('Inference failed'))
        with self.assertRaisesRegex(RuntimeError, 'Inference failed'):
            detection.detect_card(model, self.image, [], Mock())

    def test_no_border_or_degenerate_corners(self):
        self.assertIsNone(processing.document_detect(np.zeros_like(self.image)))
        with self.assertRaises(ValueError):
            processing.four_point_transform(self.image, np.zeros((4, 2)))

    def test_template_classification_and_threshold(self):
        templates = [np.flip(self.image, axis=1).copy(), self.image.copy()]
        self.assertEqual(detection.classify(self.image, templates), 1)
        self.assertIsNone(detection.classify(self.image, templates, threshold=1.0))
        self.assertIsNone(detection.classify(self.image, []))

    def test_face_mask_does_not_modify_input(self):
        box = Mock()
        box.cpu.return_value.numpy.return_value = np.array([10., 8., 20., 18.])
        # Use a small sequence wrapper for the YOLO boxes interface.
        class Boxes(list):
            @property
            def xyxy(self):
                return self
        model = Mock()
        model.predict.return_value = [SimpleNamespace(boxes=Boxes([box]))]
        original = self.image.copy()
        masked = detection.remove_face(model, self.image)
        self.assertTrue(np.all(masked[10:16, 12:18] == 0))
        np.testing.assert_array_equal(masked[0, 0], original[0, 0])
        np.testing.assert_array_equal(self.image, original)

    def test_load_image_converts_grayscale_to_rgb(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gray.png'
            Image.new('L', (4, 4), 100).save(path)
            loaded = processing.load_image(path, (8, 6))
        self.assertEqual(loaded.shape, (6, 8, 3))
        self.assertTrue(np.all(loaded == 100))

    def test_template_order_is_preserved_and_nonimages_are_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in [('2.png', 20), ('1.png', 10)]:
                Image.new('RGB', (4, 4), (value, value, value)).save(root / name)
            (root / 'notes.txt').write_text('not a template')
            face_model = Mock()
            face_model.predict.return_value = [SimpleNamespace(boxes=[])]
            with patch.object(detection.os, 'listdir', return_value=['2.png', 'notes.txt', '1.png']):
                templates = detection.load_templates(root, face_model, (4, 4))
        self.assertEqual([int(image[0, 0, 0]) for image in templates], [20, 10])


if __name__ == '__main__':
    unittest.main()
