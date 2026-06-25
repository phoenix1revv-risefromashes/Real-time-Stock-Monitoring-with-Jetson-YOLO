from collections import deque
from pathlib import Path
from datetime import datetime
import csv
import json

import cv2
import numpy as np
from ultralytics import YOLO


MODEL_DIR = Path("model")
REFERENCE_PATH = Path("configs/shelf_base_reference.json")

LOG_DIR = Path("logs")
EVENT_LOG_PATH = LOG_DIR / "live_stock_events.csv"

CAMERA_INDEX = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FRAME_FPS = 30

IMAGE_SIZE = 640
CONFIDENCE = 0.25

SHELF_CLASS = "shelf_base"
ITEM_CLASSES = ["book", "can"]

SMOOTHING_FRAMES = 10
MIN_OVERLAP_PIXELS = 100
MIN_ITEM_AREA_PIXELS = 300
MAX_ITEM_TO_SLOT_AREA_RATIO = 0.90
ROI_PADDING = 40

STOCK_PERCENT_OFFSET = 15.0
EMPTY_DEADBAND_PERCENT = 10.0

ITEM_ROI_OFFSET_X = 0
ITEM_ROI_OFFSET_Y = -120

SLOT_OVERLAY_ALPHA = 0.12


def find_model_path():
    model_files = sorted(MODEL_DIR.rglob("*.pt"))

    if not model_files:
        raise FileNotFoundError("No .pt model found inside model/")

    return model_files[0]


def create_item_roi_polygon(base_polygon):
    base_polygon = base_polygon.astype(np.int32)

    upper_polygon = base_polygon.copy()
    upper_polygon[:, 0] += ITEM_ROI_OFFSET_X
    upper_polygon[:, 1] += ITEM_ROI_OFFSET_Y

    roi_points = np.vstack([base_polygon, upper_polygon])
    roi_hull = cv2.convexHull(roi_points)

    return roi_hull.reshape(-1, 2)


def load_reference_slots():
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Missing reference file: {REFERENCE_PATH}. "
            "Run tools/enroll_shelf_base.py first."
        )

    with open(REFERENCE_PATH, "r") as file:
        reference_data = json.load(file)

    slots = []

    for slot in reference_data["slots"]:
        base_polygon = np.array(slot["polygon"], dtype=np.int32)
        item_roi_polygon = create_item_roi_polygon(base_polygon)

        slots.append(
            {
                "name": slot["name"],
                "reference_area_pixels": float(slot["reference_area_pixels"]),
                "center_x": float(slot["center_x"]),
                "center_y": float(slot["center_y"]),
                "polygon": base_polygon,
                "item_roi_polygon": item_roi_polygon,
                "reference_mask": None,
                "item_roi_mask": None,
                "current_area_pixels": 0.0,
                "smoothed_area_pixels": 0.0,
                "stock_percent": 0.0,
                "stock_status": "Unknown",
                "has_book": False,
                "has_can": False,
                "item_type": "No Items",
            }
        )

    return slots


def open_camera():
    camera = cv2.VideoCapture(CAMERA_INDEX)

    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, FRAME_FPS)

    if not camera.isOpened():
        raise RuntimeError("Could not open camera.")

    return camera


def create_polygon_mask(frame_shape, polygon):
    height, width = frame_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def prepare_reference_masks(slots, frame_shape):
    for slot in slots:
        slot["reference_mask"] = create_polygon_mask(frame_shape, slot["polygon"])
        slot["item_roi_mask"] = create_polygon_mask(frame_shape, slot["item_roi_polygon"])


def get_processing_roi(slots, frame_shape):
    height, width = frame_shape[:2]

    all_points = []

    for slot in slots:
        all_points.extend(slot["item_roi_polygon"].reshape(-1, 2))

    all_points = np.array(all_points, dtype=np.int32)

    x, y, w, h = cv2.boundingRect(all_points)

    x1 = max(0, x - ROI_PADDING)
    y1 = max(0, y - ROI_PADDING)
    x2 = min(width, x + w + ROI_PADDING)
    y2 = min(height, y + h + ROI_PADDING)

    return x1, y1, x2, y2


def crop_roi(frame, roi):
    x1, y1, x2, y2 = roi
    return frame[y1:y2, x1:x2]


def shift_polygon_to_full_frame(polygon, offset_x, offset_y):
    shifted_polygon = polygon.copy()
    shifted_polygon[:, 0] += offset_x
    shifted_polygon[:, 1] += offset_y
    return shifted_polygon


def extract_detections(result, roi_offset):
    detections = []

    if result.boxes is None or result.masks is None:
        return detections

    offset_x, offset_y = roi_offset

    for index, box in enumerate(result.boxes):
        class_id = int(box.cls[0].item())
        class_name = result.names[class_id]
        confidence = float(box.conf[0].item())

        if class_name != SHELF_CLASS and class_name not in ITEM_CLASSES:
            continue

        polygon_points = result.masks.xy[index]

        if polygon_points is None or len(polygon_points) < 3:
            continue

        polygon = np.array(polygon_points, dtype=np.int32)
        polygon = shift_polygon_to_full_frame(polygon, offset_x, offset_y)

        area_pixels = float(cv2.contourArea(polygon))

        moments = cv2.moments(polygon)

        if moments["m00"] != 0:
            center_x = float(moments["m10"] / moments["m00"])
            center_y = float(moments["m01"] / moments["m00"])
        else:
            center = polygon.mean(axis=0)
            center_x = float(center[0])
            center_y = float(center[1])

        detections.append(
            {
                "class_name": class_name,
                "confidence": confidence,
                "area_pixels": area_pixels,
                "center_x": center_x,
                "center_y": center_y,
                "polygon": polygon,
            }
        )

    return detections


def reset_live_slot_values(slots):
    for slot in slots:
        slot["current_area_pixels"] = 0.0
        slot["has_book"] = False
        slot["has_can"] = False
        slot["item_type"] = "No Items"


def match_shelf_base_to_slots(detections, slots, frame_shape):
    for detection in detections:
        if detection["class_name"] != SHELF_CLASS:
            continue

        detection_mask = create_polygon_mask(frame_shape, detection["polygon"])

        best_slot = None
        best_overlap_area = 0

        for slot in slots:
            overlap_mask = cv2.bitwise_and(detection_mask, slot["reference_mask"])
            overlap_area = int(cv2.countNonZero(overlap_mask))

            if overlap_area > best_overlap_area:
                best_overlap_area = overlap_area
                best_slot = slot

        if best_slot is not None and best_overlap_area > MIN_OVERLAP_PIXELS:
            best_slot["current_area_pixels"] += float(best_overlap_area)


def point_inside_mask(mask, x, y):
    height, width = mask.shape[:2]

    x = int(x)
    y = int(y)

    if x < 0 or y < 0 or x >= width or y >= height:
        return False

    return mask[y, x] > 0


def match_items_to_slots(detections, slots, frame_shape):
    for detection in detections:
        if detection["class_name"] not in ITEM_CLASSES:
            continue

        if detection["area_pixels"] < MIN_ITEM_AREA_PIXELS:
            continue

        detection_mask = create_polygon_mask(frame_shape, detection["polygon"])

        best_slot = None
        best_overlap_score = 0

        for slot in slots:
            center_inside = point_inside_mask(
                slot["item_roi_mask"],
                detection["center_x"],
                detection["center_y"],
            )

            overlap_mask = cv2.bitwise_and(detection_mask, slot["item_roi_mask"])
            overlap_area = int(cv2.countNonZero(overlap_mask))

            overlap_score = overlap_area

            if center_inside:
                overlap_score += 10000

            if overlap_score > best_overlap_score:
                best_overlap_score = overlap_score
                best_slot = slot

        if best_slot is None:
            continue

        if best_overlap_score <= MIN_OVERLAP_PIXELS:
            continue

        item_to_slot_area_ratio = detection["area_pixels"] / best_slot["reference_area_pixels"]

        if item_to_slot_area_ratio > MAX_ITEM_TO_SLOT_AREA_RATIO:
            continue

        if detection["class_name"] == "book":
            best_slot["has_book"] = True

        elif detection["class_name"] == "can":
            best_slot["has_can"] = True


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def get_stock_status(stock_percent):
    if stock_percent <= 20:
        return "Empty / Need Restock ASAP"

    if stock_percent <= 50:
        return "Low Stock / Restock Soon"

    if stock_percent <= 75:
        return "Partial / Light Restocking"

    if stock_percent <= 93:
        return "Almost Full / No Restocking"

    return "Full / No Restocking"


def get_item_type(has_book, has_can):
    if has_book and has_can:
        return "Mixed Items"

    if has_book:
        return "Books"

    if has_can:
        return "Cans"

    return "No Items"


def get_slot_color(slot):
    if slot["item_type"] == "Mixed Items":
        return (0, 0, 255)

    if slot["stock_status"] == "Empty / Need Restock ASAP":
        return (0, 0, 255)

    if slot["stock_status"] == "Low Stock / Restock Soon":
        return (0, 255, 255)

    if slot["stock_status"] == "Partial / Light Restocking":
        return (0, 165, 255)

    if slot["stock_status"] == "Almost Full / No Restocking":
        return (255, 0, 0)

    if slot["stock_status"] == "Full / No Restocking":
        return (0, 255, 0)

    return (255, 255, 255)


def update_stock_levels(slots, histories):
    for slot in slots:
        reference_area = slot["reference_area_pixels"]
        current_area = clamp(slot["current_area_pixels"], 0.0, reference_area)

        histories[slot["name"]].append(current_area)

        smoothed_area = sum(histories[slot["name"]]) / len(histories[slot["name"]])

        visible_ratio = smoothed_area / reference_area
        stock_ratio = 1.0 - visible_ratio
        raw_stock_percent = clamp(stock_ratio * 100.0, 0.0, 100.0)

        if raw_stock_percent <= EMPTY_DEADBAND_PERCENT:
            stock_percent = 0.0
        else:
            stock_percent = clamp(
                raw_stock_percent + STOCK_PERCENT_OFFSET,
                0.0,
                100.0,
            )

        slot["smoothed_area_pixels"] = smoothed_area
        slot["stock_percent"] = stock_percent
        slot["stock_status"] = get_stock_status(stock_percent)
        slot["item_type"] = get_item_type(slot["has_book"], slot["has_can"])


def ensure_event_log_file():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if EVENT_LOG_PATH.exists():
        return

    with open(EVENT_LOG_PATH, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "timestamp",
                "slot_name",
                "previous_stock_status",
                "current_stock_status",
                "previous_item_type",
                "current_item_type",
                "stock_percent",
                "event_type",
            ]
        )


def get_current_slot_state(slot):
    return {
        "stock_status": slot["stock_status"],
        "item_type": slot["item_type"],
        "stock_percent": slot["stock_percent"],
    }


def get_event_type(previous_state, current_state):
    stock_changed = previous_state["stock_status"] != current_state["stock_status"]
    item_changed = previous_state["item_type"] != current_state["item_type"]

    if stock_changed and item_changed:
        return "stock_and_item_change"

    if stock_changed:
        return "stock_status_change"

    if item_changed:
        return "item_type_change"

    return None


def write_event_log(slot_name, previous_state, current_state, event_type):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(EVENT_LOG_PATH, "a", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                timestamp,
                slot_name,
                previous_state["stock_status"],
                current_state["stock_status"],
                previous_state["item_type"],
                current_state["item_type"],
                f"{current_state['stock_percent']:.1f}",
                event_type,
            ]
        )


def log_slot_events(slots, previous_slot_states):
    for slot in slots:
        slot_name = slot["name"]
        current_state = get_current_slot_state(slot)

        previous_state = previous_slot_states.get(slot_name)

        if previous_state is None:
            previous_slot_states[slot_name] = current_state
            continue

        event_type = get_event_type(previous_state, current_state)

        if event_type is not None:
            write_event_log(
                slot_name,
                previous_state,
                current_state,
                event_type,
            )

            print(
                f"[EVENT] {slot_name}: "
                f"{previous_state['stock_status']} -> {current_state['stock_status']} | "
                f"{previous_state['item_type']} -> {current_state['item_type']} | "
                f"{current_state['stock_percent']:.1f}% | "
                f"{event_type}"
            )

        previous_slot_states[slot_name] = current_state


def draw_text_block(frame, lines, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.62
    thickness = 2
    line_height = 28

    for index, line in enumerate(lines):
        text_y = int(y + index * line_height)

        cv2.putText(
            frame,
            line,
            (int(x) + 2, text_y + 2),
            font,
            font_scale,
            (0, 0, 0),
            thickness + 2,
        )

        cv2.putText(
            frame,
            line,
            (int(x), text_y),
            font,
            font_scale,
            color,
            thickness,
        )


def draw_slot_results(frame, slots):
    overlay = frame.copy()

    for slot in slots:
        color = get_slot_color(slot)
        cv2.fillPoly(overlay, [slot["polygon"]], color)

    display_frame = cv2.addWeighted(
        overlay,
        SLOT_OVERLAY_ALPHA,
        frame,
        1.0 - SLOT_OVERLAY_ALPHA,
        0,
    )

    for slot in slots:
        color = get_slot_color(slot)

        cv2.polylines(display_frame, [slot["polygon"]], True, color, 3)

        lines = [
            f"{slot['name']}",
            f"Stock level: {slot['stock_percent']:.1f}% | {slot['stock_status']}",
            f"Type: {slot['item_type']}",
        ]

        x = int(slot["center_x"]) - 190
        y = int(slot["center_y"]) - 10

        draw_text_block(display_frame, lines, x, y, color)

    cv2.putText(
        display_frame,
        "Live Stock + Mixed Item Detection | q: quit",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 255),
        2,
    )

    return display_frame


def main():
    model_path = find_model_path()
    model = YOLO(str(model_path))

    slots = load_reference_slots()

    histories = {
        slot["name"]: deque(maxlen=SMOOTHING_FRAMES)
        for slot in slots
    }

    previous_slot_states = {}

    ensure_event_log_file()

    camera = open_camera()

    reference_masks_ready = False
    processing_roi = None

    print(f"Using model: {model_path}")
    print(f"Using reference: {REFERENCE_PATH}")
    print(f"Logging events to: {EVENT_LOG_PATH}")
    print("Processing only enrolled shelf/item ROI.")
    print("Press 'q' to quit.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read frame.")
            break

        if not reference_masks_ready:
            prepare_reference_masks(slots, frame.shape)
            processing_roi = get_processing_roi(slots, frame.shape)
            reference_masks_ready = True

        roi_frame = crop_roi(frame, processing_roi)

        results = model.predict(
            source=roi_frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            verbose=False,
        )

        roi_x1, roi_y1, _, _ = processing_roi

        detections = extract_detections(results[0], (roi_x1, roi_y1))

        reset_live_slot_values(slots)
        match_shelf_base_to_slots(detections, slots, frame.shape)
        match_items_to_slots(detections, slots, frame.shape)
        update_stock_levels(slots, histories)
        log_slot_events(slots, previous_slot_states)

        display_frame = draw_slot_results(frame, slots)

        cv2.imshow("Live Stock Level Test", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()