#!/usr/bin/env python3
"""Convert YOLO11 ONNX models to RKNN format (FP16)."""

import sys
from pathlib import Path
from rknn.api import RKNN


def convert(onnx_path: str, output_path: str, platform: str = "rk3588") -> int:
    """Convert ONNX model to RKNN format.

    Args:
        onnx_path: Path to input ONNX model.
        output_path: Path to output RKNN model.
        platform: Target platform name.

    Returns:
        0 on success, non-zero on failure.
    """
    print(f"Converting: {onnx_path}")
    print(f"Output:     {output_path}")
    print(f"Platform:   {platform}")
    print(f"Quantize:   No (FP16)")
    print()

    rknn = RKNN(verbose=False)

    print("--> Config model")
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=platform,
    )
    print("done")

    print("--> Loading model")
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print(f"ERROR: Load model failed! (ret={ret})")
        return ret
    print("done")

    print("--> Building model (FP16)")
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print(f"ERROR: Build model failed! (ret={ret})")
        return ret
    print("done")

    print("--> Export rknn model")
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print(f"ERROR: Export rknn model failed! (ret={ret})")
        return ret
    print("done")

    rknn.release()
    print(f"SUCCESS: {output_path}\n")
    return 0


def main() -> int:
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    downloads = Path.home() / "下载"

    tasks = [
        (str(downloads / "yolo11s.onnx"), str(models_dir / "yolo11s.rknn")),
        (str(downloads / "yolo11m.onnx"), str(models_dir / "yolo11m.rknn")),
    ]

    for onnx_path, rknn_path in tasks:
        if not Path(onnx_path).exists():
            print(f"SKIP: {onnx_path} not found")
            continue
        ret = convert(onnx_path, rknn_path)
        if ret != 0:
            return ret

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
