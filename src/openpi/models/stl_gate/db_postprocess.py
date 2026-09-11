# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# Adapted from PaddleX 3.7.2 text_detection/processors.py: quad/fast path only.
"""DB quadrilateral decoding, in original-image pixel coordinates."""

import math

import cv2
import numpy as np
import pyclipper


def _mini_box(contour: np.ndarray) -> tuple[np.ndarray, float]:
    rect = cv2.minAreaRect(contour)
    points = sorted(cv2.boxPoints(rect), key=lambda point: point[0])
    left = sorted(points[:2], key=lambda point: point[1])
    right = sorted(points[2:], key=lambda point: point[1])
    return np.array([left[0], right[0], right[1], left[1]]), min(rect[1])


def _box_score(probability: np.ndarray, box: np.ndarray) -> float:
    height, width = probability.shape
    x0 = max(0, min(math.floor(box[:, 0].min()), width - 1))
    x1 = max(0, min(math.ceil(box[:, 0].max()), width - 1))
    y0 = max(0, min(math.floor(box[:, 1].min()), height - 1))
    y1 = max(0, min(math.ceil(box[:, 1].max()), height - 1))
    mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
    local_box = box - np.array([x0, y0])
    cv2.fillPoly(mask, local_box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(probability[y0 : y1 + 1, x0 : x1 + 1], mask)[0]


def decode_boxes(
    probability: np.ndarray,
    image_size: tuple[int, int],
    *,
    thresh: float,
    box_thresh: float,
    unclip_ratio: float,
    max_candidates: int,
) -> tuple[np.ndarray, np.ndarray]:
    width, height = image_size
    map_height, map_width = probability.shape
    contours, _ = cv2.findContours(
        (probability > thresh).astype(np.uint8) * 255,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    boxes, scores = [], []
    for contour in contours[:max_candidates]:
        box, short_side = _mini_box(contour)
        if short_side < 3:
            continue
        score = _box_score(probability, box)
        if score < box_thresh:
            continue
        perimeter = cv2.arcLength(box, closed=True)
        if perimeter <= 0:
            continue
        distance = cv2.contourArea(box) * unclip_ratio / perimeter
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(distance)
        if not expanded:
            continue
        # The reference keeps the first path if multiple paths have unequal lengths.
        try:
            expanded = np.array(expanded, dtype=np.int32)
        except ValueError:
            expanded = np.array(expanded[0], dtype=np.int32)
        box, short_side = _mini_box(expanded.reshape(-1, 1, 2))
        if short_side < 5:
            continue
        for point in box:
            point[0] = max(0, min(round(point[0] * (width / map_width)), width))
            point[1] = max(0, min(round(point[1] * (height / map_height)), height))
        boxes.append(box.astype(np.int32))
        scores.append(score)
    return np.asarray(boxes, dtype=np.int32).reshape(-1, 4, 2), np.asarray(scores, dtype=np.float32)
