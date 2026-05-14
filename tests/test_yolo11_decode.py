#!/usr/bin/env python3
"""Test YOLO11 decode logic with synthetic 9-output data."""

import numpy as np
import sys
sys.path.insert(0, '.')

from app.infer.engine import InferenceEngine


def test_yolo11_decode():
    """Test YOLO11 output format detection and decoding."""
    class_names = [f"class_{i}" for i in range(80)]
    engine = InferenceEngine(
        model_path="dummy",
        class_names=class_names,
        conf_threshold=0.25,
        nms_threshold=0.45,
        input_size=640,
    )

    # Create synthetic YOLO11 outputs (9 outputs, 3 scales)
    outputs = []
    for scale_h, scale_w in [(80, 80), (40, 40), (20, 20)]:
        box_output = np.random.randn(1, 64, scale_h, scale_w).astype(np.float32) * 2
        cls_output = np.random.uniform(0.001, 0.01, (1, 80, scale_h, scale_w)).astype(np.float32)
        conf_output = np.random.uniform(0.1, 0.3, (1, 1, scale_h, scale_w)).astype(np.float32)
        outputs.extend([box_output, cls_output, conf_output])

    # Inject strong detection at scale 0 (80x80), position (40, 40)
    outputs[1][0, 5, 40, 40] = 0.8  # class 5 with high confidence
    outputs[1][0, 10, 20, 20] = 0.6  # class 10

    print("Testing YOLO11 format detection (9 outputs)...")
    result = engine._try_decode_yolo11(outputs, ts_ms=1000)

    if result is not None:
        print(f"SUCCESS: Detected YOLO11 format, got {len(result)} detections")
        for det in result[:5]:
            print(f"  - {det.class_name}: conf={det.conf:.3f}, bbox={det.bbox_xyxy}")
    else:
        print("FAIL: YOLO11 format not detected")

    # Test with 3 outputs (should return None)
    print("\nTesting with 3 outputs (should return None)...")
    result2 = engine._try_decode_yolo11(outputs[:3], ts_ms=1000)
    if result2 is None:
        print("SUCCESS: Correctly returned None for 3-output format")
    else:
        print("FAIL: Should have returned None")

    return 0


if __name__ == "__main__":
    raise SystemExit(test_yolo11_decode())
