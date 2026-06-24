from datetime import datetime
from pathlib import Path
import json

import cv2
import numpy as np
from ultralytics import YOLO


MODEL_DIR = Path("model")
REFERENCE_PATH = Path("configs/shelf_base_reference.json")

CAMERA_INDEX = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FRAME_FPS = 30

IMAGE_SIZE = 640
CONFIDENCE = 0.25
TARGET_CLASS_NAME = "shelf_base"


def find_model_path():
    model_files = sorted(MODEL_DIR.rglob("*.pt"))

    if not model_files:
        raise FileNotFoundError("No .pt model found inside model/")

    return model_files[0]


def open_camera():
    camera = cv2.VideoCapture(CAMERA_INDEX)

    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, FRAME_FPS)

    if not camera.isOpened():
        raise RuntimeError("Could not open camera.")

    return camera


def extract_shelf_base_detections(result):
    detections = []

    if result.boxes is None or result.masks is None:
        return detections

    for index, box in enumerate(result.boxes):
        class_id = int(box.cls[0].item())
        class_name = result.names[class_id]
        confidence = float(box.conf[0].item())

        if class_name != TARGET_CLASS_NAME:
            continue

        polygon_points = result.masks.xy[index]

        if polygon_points is None or len(polygon_points) < 3:
            continue

        polygon = np.array(polygon_points, dtype=np.int32)
        area_pixels = float(cv2.contourArea(polygon))

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append(
            {
                "slot_name": "",
                "confidence": confidence,
                "area_pixels": area_pixels,
                "center_x": float((x1 + x2) / 2),
                "center_y": float((y1 + y2) / 2),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "polygon": polygon,
            }
        )

    detections.sort(key=lambda item: (item["center_y"], item["center_x"]))

    for index, detection in enumerate(detections, start=1):
        detection["slot_name"] = f"S{index}"

    return detections


def draw_overlay(frame, detections):
    overlay = frame.copy()

    for detection in detections:
        cv2.fillPoly(overlay, [detection["polygon"]], (0, 255, 255))

    display_frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    for detection in detections:
        cv2.polylines(display_frame, [detection["polygon"]], True, (0, 255, 255), 2)

        label = (
            f"{detection['slot_name']} | "
            f"{detection['area_pixels']:.0f}px | "
            f"{detection['confidence']:.2f}"
        )

        cv2.putText(
            display_frame,
            label,
            (int(detection["center_x"]) - 160, int(detection["center_y"])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

    cv2.putText(
        display_frame,
        "Enrollment Mode | e: save | q: quit",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )

    return display_frame


def save_reference(detections, model_path):
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)

    reference_data = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": str(model_path),
        "camera": {
            "index": CAMERA_INDEX,
            "width": FRAME_WIDTH,
            "height": FRAME_HEIGHT,
            "fps": FRAME_FPS,
        },
        "reference_class": TARGET_CLASS_NAME,
        "slots": [],
    }

    for detection in detections:
        reference_data["slots"].append(
            {
                "name": detection["slot_name"],
                "reference_area_pixels": detection["area_pixels"],
                "confidence": detection["confidence"],
                "center_x": detection["center_x"],
                "center_y": detection["center_y"],
                "bbox": detection["bbox"],
                "polygon": detection["polygon"].tolist(),
            }
        )

    with open(REFERENCE_PATH, "w") as file:
        json.dump(reference_data, file, indent=2)

    print(f"Saved: {REFERENCE_PATH}")
    print(f"Slots saved: {len(detections)}")


def main():
    model_path = find_model_path()
    model = YOLO(str(model_path))
    camera = open_camera()

    print(f"Using model: {model_path}")
    print("Press 'e' to save shelf-base reference.")
    print("Press 'q' to quit.")

    latest_detections = []

    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read frame.")
            break

        results = model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            verbose=False,
        )

        latest_detections = extract_shelf_base_detections(results[0])
        display_frame = draw_overlay(frame, latest_detections)

        cv2.imshow("Shelf Base Enrollment", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("e"):
            if latest_detections:
                save_reference(latest_detections, model_path)
            else:
                print("No shelf_base detections found.")

        elif key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()