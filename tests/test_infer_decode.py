from __future__ import annotations

import numpy as np

from app.infer.engine import InferenceEngine


def _engine() -> InferenceEngine:
    return InferenceEngine(
        model_path="./models/detector.rknn",
        class_names=["cup", "phone", "keyboard", "scissors"],
        conf_threshold=0.35,
        nms_threshold=0.45,
        input_size=640,
    )


def _assert_valid(dets):
    assert dets
    for d in dets:
        assert 0.0 <= d.conf <= 1.0
        x1, y1, x2, y2 = d.bbox_xyxy
        assert 0 <= x1 < x2 <= 639
        assert 0 <= y1 < y2 <= 639


def test_decode_nx6_with_conf_out_of_range_and_normalized_boxes():
    eng = _engine()
    out = np.array(
        [
            [0.10, 0.10, 0.42, 0.40, 1.8, 1],
            [0.50, 0.45, 0.18, 0.16, 0.9, 0],
        ],
        dtype=np.float32,
    )
    dets = eng._decode([out], 1)
    _assert_valid(dets)


def test_decode_yolo_with_obj_and_logit_scores():
    eng = _engine()
    # [cx, cy, w, h, obj, cls0, cls1, cls2, cls3]
    out = np.array(
        [
            [0.52, 0.48, 0.24, 0.20, 5.0, -4.0, 6.0, -4.0, -4.0],
            [0.30, 0.35, 0.20, 0.25, -6.0, 7.0, -3.0, -3.0, -3.0],
        ],
        dtype=np.float32,
    )
    dets = eng._decode([out], 2)
    _assert_valid(dets)
    assert any(d.class_name == "phone" for d in dets)


def test_decode_split_outputs_boxes_and_classes():
    eng = _engine()
    boxes = np.array([[[0.2, 0.2, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2]]], dtype=np.float32)
    classes = np.array([[[8.0, -5.0, -5.0, -5.0], [-5.0, -5.0, -5.0, 7.5]]], dtype=np.float32)
    dets = eng._decode([boxes, classes], 3)
    _assert_valid(dets)
    names = {d.class_name for d in dets}
    assert "cup" in names
    assert "scissors" in names
