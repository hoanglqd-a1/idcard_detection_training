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

    def detected_model(self):
        corners = Mock()
        corners.cpu.return_value.numpy.return_value = np.array(
            [[0., 0.], [63., 0.], [63., 31.], [0., 31.]],
        )
        confidence = Mock()
        confidence.item.return_value = 0.95

        class OBB:
            xyxyxyxy = [corners]
            conf = [confidence]

            def __len__(self):
                return 1

        return Mock(return_value=[SimpleNamespace(obb=OBB())])

    def test_rich_no_detection_result(self):
        model = Mock(return_value=[SimpleNamespace(obb=[])])
        result = detection.analyze_card(model, self.image, [], Mock())
        self.assertEqual(result.status, detection.DetectionStatus.NO_CARD)
        self.assertFalse(result.card_detected)
        self.assertIsNone(result.is_supported)
        self.assertIsNone(result.card)
        self.assertIsNone(result.corners)
        self.assertIsNone(result.detection_confidence)
        self.assertIsNone(result.match_score)
        self.assertGreaterEqual(result.processing_time_ms, 0)

    def test_extraction_failures_preserve_detection(self):
        for stage in ('crop', 'refinement'):
            with self.subTest(stage=stage):
                with patch.object(detection, 'crop_image') as crop, patch.object(
                    detection, 'document_detect', return_value=None,
                ):
                    crop.return_value = self.image
                    if stage == 'crop':
                        crop.side_effect = ValueError('Degenerate corners')
                    result = detection.analyze_card(
                        self.detected_model(), self.image, [], Mock(),
                    )
                self.assertEqual(result.status, detection.DetectionStatus.EXTRACTION_FAILED)
                self.assertEqual(result.failure_stage, stage)
                self.assertTrue(result.card_detected)
                self.assertEqual(result.detection_confidence, 0.95)
                self.assertEqual(result.corners.shape, (4, 2))
                self.assertIsNone(result.is_supported)
                self.assertIsNone(result.match_score)

    def test_matching_results_and_legacy_wrapper_agree(self):
        # Exercise real OpenCV correlation; isolate geometric refinement only.
        face_model = Mock()
        face_model.predict.return_value = [SimpleNamespace(boxes=[])]
        card = detection.cv2.resize(self.image, detection.CARD_SIZE)
        templates = [np.flip(card, axis=1).copy(), card.copy()]
        for threshold, matched in ((0.8, True), (1.0, False)):
            with self.subTest(threshold=threshold), patch.object(
                detection, 'document_detect', return_value=card,
            ):
                model = self.detected_model()
                result = detection.analyze_card(
                    model, self.image, templates, face_model, threshold,
                )
                model.assert_called_once_with(self.image)
                legacy_card, legacy_label = detection.detect_card(
                    model, self.image, templates, face_model, threshold,
                )
                self.assertEqual(result.is_supported, matched)
                self.assertEqual(result.status, detection.DetectionStatus.MATCHED
                                 if matched else detection.DetectionStatus.UNMATCHED)
                self.assertEqual(result.template_index, 1 if matched else None)
                self.assertAlmostEqual(result.match_score, 1.0, places=5)
                self.assertEqual(result.detection_confidence, 0.95)
                np.testing.assert_array_equal(result.card, card)
                np.testing.assert_array_equal(legacy_card, result.card)
                self.assertEqual(legacy_label, result.template_index)

    def test_empty_templates_have_no_score(self):
        self.assertEqual(detection.classify_with_score(self.image, []), (None, None))

    def test_default_threshold_rejects_scores_at_or_below_point_eight(self):
        for score, expected in ((0.75, None), (0.8, None), (0.81, 0)):
            with self.subTest(score=score), patch.object(
                detection, 'match', return_value=np.array([score]),
            ):
                self.assertEqual(detection.classify(self.image, [self.image]), expected)
                self.assertEqual(
                    detection.classify_with_score(self.image, [self.image]),
                    (expected, score),
                )

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
            with patch.object(detection.os, 'listdir', return_value=['2.png', 'notes.txt', '1.png']):
                templates = detection.load_templates(root, card_size=(4, 4))
        self.assertEqual([int(image[0, 0, 0]) for image in templates], [20, 10])


if __name__ == '__main__':
    unittest.main()
