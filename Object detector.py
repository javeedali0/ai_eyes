"""
ObjectScan - Python object detector
------------------------------------
Detects objects via your webcam (or a photo) and labels them with their names.

SETUP (run once):
    pip install ultralytics opencv-python

USAGE:
    python object_detector.py                # live webcam
    python object_detector.py photo.jpg       # detect on a single image
    python object_detector.py --camera 1      # use a different camera index

Controls (webcam mode):
    q  -> quit
    s  -> save a snapshot of the current frame to disk

Notes:
- First run downloads a small pretrained model (yolov8n.pt, ~6MB) automatically.
- Recognizes 80 common categories (people, animals, vehicles, everyday objects, etc.)
  from the COCO dataset. It won't have a name for everything, and accuracy depends
  on lighting, angle, and distance -- like any detector, it isn't infallible.
"""

import sys
import argparse
import time
from pathlib import Path

def die(msg):
    print(f"\n[ERROR] {msg}\n")
    sys.exit(1)

try:
    import cv2
except ImportError:
    die("OpenCV is not installed. Run:  pip install opencv-python")

try:
    from ultralytics import YOLO
except ImportError:
    die("Ultralytics is not installed. Run:  pip install ultralytics")


def load_model(model_name="yolov8n.pt"):
    print(f"Loading model ({model_name})... first run may download it, please wait.")
    model = YOLO(model_name)
    print("Model ready.")
    return model


def draw_detections(frame, result):
    """Draw boxes + labels on frame, return sorted list of (name, confidence)."""
    names = result.names
    found = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = names[cls_id]
        found.append((label, conf))

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = (0, 255, 150)  # BGR, green-ish
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, text, (x1 + 3, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 20, 15), 2)

    found.sort(key=lambda x: -x[1])
    return found


def print_names(found):
    if not found:
        print("  No recognizable objects found.")
        return
    seen = {}
    for name, conf in found:
        if name not in seen or conf > seen[name]:
            seen[name] = conf
    for name, conf in sorted(seen.items(), key=lambda kv: -kv[1]):
        print(f"  - {name} ({conf*100:.0f}%)")


def run_on_image(model, image_path, conf_threshold):
    path = Path(image_path)
    if not path.exists():
        die(f"Image not found: {image_path}")

    frame = cv2.imread(str(path))
    if frame is None:
        die(f"Could not read image (unsupported format?): {image_path}")

    result = model.predict(frame, conf=conf_threshold, verbose=False)[0]
    found = draw_detections(frame, result)

    print(f"\nObjects found in {path.name}:")
    print_names(found)

    out_path = path.with_name(path.stem + "_detected" + path.suffix)
    cv2.imwrite(str(out_path), frame)
    print(f"\nSaved labeled image to: {out_path}")

    cv2.imshow("ObjectScan - press any key to close", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_on_webcam(model, camera_index, conf_threshold):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        die(f"Could not open camera index {camera_index}. "
            f"Try a different --camera number, or check camera permissions.")

    print("\nWebcam started. Press 'q' to quit, 's' to save a snapshot.\n")
    last_print = 0
    snap_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Lost camera feed, stopping.")
            break

        result = model.predict(frame, conf=conf_threshold, verbose=False)[0]
        found = draw_detections(frame, result)

        cv2.putText(frame, "Press 'q' to quit, 's' to snapshot", (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 220), 1)
        cv2.imshow("ObjectScan (Python)", frame)

        # print names to console at most twice a second, to keep it readable
        now = time.time()
        if now - last_print > 0.5:
            print("\rCurrently seeing: " + (", ".join(sorted({n for n, _ in found})) or "nothing recognizable") + " " * 10, end="")
            last_print = now

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            snap_count += 1
            snap_name = f"snapshot_{snap_count}.jpg"
            cv2.imwrite(snap_name, frame)
            print(f"\nSaved {snap_name}")

    cap.release()
    cv2.destroyAllWindows()
    print("\nStopped.")


def main():
    parser = argparse.ArgumentParser(description="Detect and name objects via webcam or photo.")
    parser.add_argument("image", nargs="?", help="Path to an image file. Omit to use the webcam.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold 0-1 (default: 0.4)")
    parser.add_argument("--model", default="yolov8n.pt", help="Model weights (default: yolov8n.pt)")
    args = parser.parse_args()

    model = load_model(args.model)

    if args.image:
        run_on_image(model, args.image, args.conf)
    else:
        run_on_webcam(model, args.camera, args.conf)


if __name__ == "__main__":
    main()
