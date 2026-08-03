"""
Run inference with the trained aircraft-detection model on an image, video,
or webcam feed.

Usage:
    python scripts/detect.py --source test_files/image.jpg
    python scripts/detect.py --source test_files/video.mp4
    python scripts/detect.py --source 0                       # webcam
    python scripts/detect.py --source video.mp4 --play        # auto-open result
    python scripts/detect.py --source image.jpg --conf 0.5
"""
import argparse
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "models" / "best.pt"
OUTPUT_DIR = ROOT / "outputs"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def resolve_source_kind(source: str) -> str:
    if source.isdigit():
        return "webcam"
    ext = Path(source).suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    raise ValueError(f"Unrecognized source: {source}. Use an image/video path or a webcam index like 0.")


def open_file(path: Path):
    """Best-effort cross-platform 'open this file' for --play."""
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["open", str(path)], check=False)
        elif system == "Windows":
            import os
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception as e:
        print(f"Could not auto-open {path}: {e}")


def run_image(model, source, conf, out_dir):
    results = model.predict(source=source, conf=conf, save=True, project=str(out_dir), name="predict", exist_ok=True)
    r = results[0]
    print(f"Detected {len(r.boxes)} aircraft.")
    for box in r.boxes:
        cls_name = model.names[int(box.cls)]
        print(f"  {cls_name:14s} conf={float(box.conf):.2f}")
    saved_path = Path(r.save_dir) / Path(source).name
    return saved_path


def run_video(model, source, conf, out_dir):
    results = model.predict(source=source, conf=conf, save=True, project=str(out_dir), name="predict",
                             exist_ok=True, stream=True)
    n_frames, t0 = 0, time.time()
    save_dir = None
    for r in results:
        n_frames += 1
        save_dir = r.save_dir
    elapsed = time.time() - t0
    fps = n_frames / elapsed if elapsed > 0 else 0
    print(f"Processed {n_frames} frames in {elapsed:.1f}s ({fps:.1f} FPS).")
    saved_path = Path(save_dir) / Path(str(source)).name if save_dir else None
    return saved_path


def run_webcam(model, cam_index, conf):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {cam_index}.")

    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = model.predict(frame, conf=conf, verbose=False)
        annotated = results[0].plot()
        cv2.imshow("Military Aircraft Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True,
                         help="Image path, video path, or webcam index (e.g. 0)")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--play", action="store_true", help="Auto-open the result after processing")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        sys.exit(
            f"No weights found at {weights_path}.\n"
            "Train a model first (`python scripts/train.py`) or point --weights at a checkpoint."
        )

    model = YOLO(str(weights_path))
    kind = resolve_source_kind(args.source)
    OUTPUT_DIR.mkdir(exist_ok=True)

    if kind == "webcam":
        run_webcam(model, int(args.source), args.conf)
        return
    elif kind == "image":
        saved_path = run_image(model, args.source, args.conf, OUTPUT_DIR)
    else:
        saved_path = run_video(model, args.source, args.conf, OUTPUT_DIR)

    if saved_path and saved_path.exists():
        print(f"Saved result: {saved_path}")
        if args.play:
            open_file(saved_path)
    elif saved_path:
        print(f"Result expected at {saved_path} but wasn't found -- check the predict/ output folder.")


if __name__ == "__main__":
    main()
