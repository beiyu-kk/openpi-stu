import cv2
import numpy as np
import pytest

from openpi.models.stl_gate import DBTextDetector
from openpi.models.stl_gate.db_postprocess import decode_boxes


def _decode(probability, image_size):
    return decode_boxes(probability, image_size, thresh=0.2, box_thresh=0.4, unclip_ratio=1.4, max_candidates=3000)


def test_empty_detection_shapes():
    polygons, scores = _decode(np.zeros((32, 64), np.float32), (640, 320))
    assert polygons.shape == (0, 4, 2)
    assert polygons.dtype == np.int32
    assert scores.shape == (0,)


def test_rotated_region_maps_back_to_source():
    probability = np.zeros((100, 160), np.float32)
    points = cv2.boxPoints(((80, 50), (50, 15), 25)).astype(np.int32)
    cv2.fillPoly(probability, [points], 0.9)
    polygons, scores = _decode(probability, (640, 400))
    assert polygons.shape == (1, 4, 2)
    assert scores[0] > 0.7
    assert np.all(polygons >= 0)
    assert np.all(polygons <= [640, 400])
    center = tuple((points.mean(axis=0) * 4).tolist())
    assert cv2.pointPolygonTest(polygons[0].astype(np.float32), center, measureDist=False) >= 0
    assert cv2.contourArea(polygons[0]) > cv2.contourArea(points) * 16


def test_low_score_regions_are_rejected():
    probability = np.full((40, 80), 0.3, dtype=np.float32)
    polygons, scores = _decode(probability, (800, 400))
    assert len(polygons) == len(scores) == 0


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((3, 32, 32), np.uint8),
        np.zeros((32, 32, 3), np.float32),
        np.zeros((32, 32), np.uint8),
        np.zeros((0, 32, 3), np.uint8),
    ],
)
def test_bad_inputs_fail_before_inference(image):
    detector = object.__new__(DBTextDetector)
    detector._predictor = object()  # noqa: SLF001
    with pytest.raises(ValueError, match="Expected|empty"):
        detector.detect(image)


def test_close_is_idempotent_and_prevents_inference():
    detector = object.__new__(DBTextDetector)
    detector.close()
    detector.close()
    with pytest.raises(RuntimeError, match="closed"):
        detector.detect(np.zeros((32, 32, 3), np.uint8))


def test_missing_model_fails_before_paddle_load(tmp_path):
    with pytest.raises(FileNotFoundError, match="inference.json"):
        DBTextDetector(tmp_path)
