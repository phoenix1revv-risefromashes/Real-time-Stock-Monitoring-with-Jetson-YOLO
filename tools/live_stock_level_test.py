from pathlib import Path
import json
import time
from collections import deque
import statistics

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

ITEM_CLASSES = ["book", "can"]

MIN_OVERLAP_PIXELS = 100
MIN_ITEM_AREA_PIXELS = 300
MAX_ITEM_TO_SLOT_AREA_RATIO = 0.90
ROI_PADDING = 40

<<<<<<< Updated upstream
STOCK_PERCENT_OFFSET = 10.0
EMPTY_DEADBAND_PERCENT = 10.0

=======
>>>>>>> Stashed changes
ITEM_ROI_OFFSET_X = 0
ITEM_ROI_OFFSET_Y = -120

SLOT_OVERLAY_ALPHA = 0.12
BLINK_INTERVAL_SECONDS = 0.5

COUNT_DEBUG = True
DEBUG_EVERY_N_FRAMES = 30

# ── Temporal smoothing ────────────────────────────────────────────────────────
# How many recent frames to keep per slot, per item class.
# Larger  → smoother but slower to react to real changes.
# Smaller → reacts faster but more flicker.
SMOOTHING_WINDOW = 15        # frames
# ─────────────────────────────────────────────────────────────────────────────


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


def read_max_item_capacity(slot):
    if "max_item_capacity" not in slot:
        raise KeyError(
            f"Slot {slot['name']} is missing max_item_capacity. "
            "Run: python3 tools/enroll_slot_capacity.py"
        )

    max_item_capacity = int(slot["max_item_capacity"])

    if max_item_capacity <= 0:
        raise ValueError(
            f"Slot {slot['name']} has invalid max_item_capacity: "
            f"{max_item_capacity}"
        )

    return max_item_capacity


def load_reference_slots():
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Missing reference file: {REFERENCE_PATH}. "
            "Run shelf-base enrollment first."
        )

    with open(REFERENCE_PATH, "r") as file:
        reference_data = json.load(file)

    slots = []

    for slot in reference_data["slots"]:
        base_polygon = np.array(slot["polygon"], dtype=np.int32)
        item_roi_polygon = create_item_roi_polygon(base_polygon)
        max_item_capacity = read_max_item_capacity(slot)

        slots.append(
            {
                "name": slot["name"],
                "reference_area_pixels": float(slot["reference_area_pixels"]),
                "center_x": float(slot["center_x"]),
                "center_y": float(slot["center_y"]),
                "polygon": base_polygon,
                "item_roi_polygon": item_roi_polygon,
                "item_roi_mask": None,
                "max_item_capacity": max_item_capacity,
                "stock_percent": 0.0,
                "stock_status": "Unknown",
                "has_book": False,
                "has_can": False,
                "book_count": 0,
                "can_count": 0,
                "total_item_count": 0,
                "item_type": "No Items",
                # Rolling history buffers — one deque per item class
                "_book_history": deque(maxlen=SMOOTHING_WINDOW),
                "_can_history": deque(maxlen=SMOOTHING_WINDOW),
            }
        )

    return slots


def print_slot_capacity_summary(slots):
    print()
    print("Loaded slot capacities:")

    for slot in slots:
        print(f"  {slot['name']}: max capacity = {slot['max_item_capacity']}")

    print()


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

        if class_name not in ITEM_CLASSES:
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
    """Reset only the raw per-frame counts; history buffers are preserved."""
    for slot in slots:
        slot["has_book"] = False
        slot["has_can"] = False
        slot["book_count"] = 0
        slot["can_count"] = 0
        slot["total_item_count"] = 0
        slot["item_type"] = "No Items"


def point_inside_mask(mask, x, y):
    height, width = mask.shape[:2]

    x = int(x)
    y = int(y)

    if x < 0 or y < 0 or x >= width or y >= height:
        return False

    return mask[y, x] > 0


def match_items_to_slots(detections, slots, frame_shape):
    for detection in detections:
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
            best_slot["book_count"] += 1

        elif detection["class_name"] == "can":
            best_slot["can_count"] += 1

        best_slot["total_item_count"] = best_slot["book_count"] + best_slot["can_count"]


def smooth_slot_counts(slots):
    """
    Push this frame's raw counts into each slot's history deque, then
    replace the live counts with the median over the window..
    """
    for slot in slots:
        slot["_book_history"].append(slot["book_count"])
        slot["_can_history"].append(slot["can_count"])

        smoothed_books = round(statistics.median(slot["_book_history"]))
        smoothed_cans  = round(statistics.median(slot["_can_history"]))

        slot["book_count"]       = smoothed_books
        slot["can_count"]        = smoothed_cans
        slot["total_item_count"] = smoothed_books + smoothed_cans
        slot["has_book"]         = smoothed_books > 0
        slot["has_can"]          = smoothed_cans  > 0


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
        return (0, 0, 255)        # Red

    if slot["stock_status"] == "Empty / Need Restock ASAP":
        return (0, 0, 255)        # Red

    if slot["stock_status"] == "Low Stock / Restock Soon":
        return (0, 255, 255)      # Yellow

    if slot["stock_status"] == "Partial / Light Restocking":
        return (0, 165, 255)      # Orange

    if slot["stock_status"] == "Almost Full / No Restocking":
        return (255, 0, 0)        # Blue

    if slot["stock_status"] == "Full / No Restocking":
        return (0, 255, 0)        # Green

    return (255, 255, 255)        # White fallback


def slot_should_blink_red(slot):
    if slot["item_type"] == "Mixed Items":
        return True

    if slot["stock_status"] == "Empty / Need Restock ASAP":
        return True

    if slot["stock_status"] == "Low Stock / Restock Soon":
        return True

    return False


def get_display_color(slot):
    if slot_should_blink_red(slot):
        blink_on = int(time.time() / BLINK_INTERVAL_SECONDS) % 2 == 0

        if blink_on:
            return (0, 0, 255)

        return (45, 45, 45)

    return get_slot_color(slot)


def update_stock_levels(slots):
    for slot in slots:
        max_capacity = max(1, int(slot["max_item_capacity"]))
        total_items = int(slot["total_item_count"])

        stock_percent = clamp(
            (total_items / max_capacity) * 100.0,
            0.0,
            100.0,
        )

        slot["stock_percent"] = stock_percent
        slot["stock_status"] = get_stock_status(stock_percent)
        slot["item_type"] = get_item_type(slot["has_book"], slot["has_can"])


<<<<<<< Updated upstream
=======
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
                "previous_item_count",
                "current_item_count",
                "max_item_capacity",
                "stock_percent",
                "event_type",
            ]
        )


def get_current_slot_state(slot):
    return {
        "stock_status": slot["stock_status"],
        "item_type": slot["item_type"],
        "total_item_count": slot["total_item_count"],
        "max_item_capacity": slot["max_item_capacity"],
        "stock_percent": slot["stock_percent"],
    }


def get_event_type(previous_state, current_state):
    stock_changed = previous_state["stock_status"] != current_state["stock_status"]
    item_changed = previous_state["item_type"] != current_state["item_type"]
    count_changed = previous_state["total_item_count"] != current_state["total_item_count"]

    if stock_changed and item_changed:
        return "stock_and_item_change"

    if stock_changed:
        return "stock_status_change"

    if item_changed:
        return "item_type_change"

    if count_changed:
        return "item_count_change"

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
                previous_state["total_item_count"],
                current_state["total_item_count"],
                current_state["max_item_capacity"],
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
                f"{previous_state['total_item_count']} -> {current_state['total_item_count']} items | "
                f"{current_state['stock_percent']:.1f}% | "
                f"{event_type}"
            )

        previous_slot_states[slot_name] = current_state


def print_count_debug(frame_index, detections, slots):
    if not COUNT_DEBUG:
        return

    if frame_index % DEBUG_EVERY_N_FRAMES != 0:
        return

    print()
    print(f"[COUNT DEBUG] Frame {frame_index}")
    print(f"YOLO visible item instances detected: {len(detections)}")

    for detection in detections:
        print(
            f"  detection={detection['class_name']} "
            f"conf={detection['confidence']:.2f} "
            f"area={detection['area_pixels']:.1f} "
            f"center=({detection['center_x']:.0f}, {detection['center_y']:.0f})"
        )

    for slot in slots:
        print(
            f"  {slot['name']}: "
            f"items={slot['total_item_count']} / {slot['max_item_capacity']} "
            f"books={slot['book_count']} "
            f"cans={slot['can_count']} "
            f"stock={slot['stock_percent']:.1f}% "
            f"type={slot['item_type']}"
        )


>>>>>>> Stashed changes
def draw_text_block(frame, lines, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.58
    thickness = 2
    line_height = 27

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
        color = get_display_color(slot)
        cv2.fillPoly(overlay, [slot["polygon"]], color)

    display_frame = cv2.addWeighted(
        overlay,
        SLOT_OVERLAY_ALPHA,
        frame,
        1.0 - SLOT_OVERLAY_ALPHA,
        0,
    )

    for slot in slots:
        color = get_display_color(slot)

        cv2.polylines(display_frame, [slot["polygon"]], True, color, 3)

        lines = [
            f"{slot['name']}",
            f"Stock level: {slot['stock_percent']:.1f}% | {slot['stock_status']}",
            f"Type: {slot['item_type']}",
            f"Items: {slot['total_item_count']} / {slot['max_item_capacity']}",
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
    print_slot_capacity_summary(slots)

    camera = open_camera()

    reference_masks_ready = False
    processing_roi = None
    frame_index = 0

    print(f"Using model: {model_path}")
    print(f"Using reference: {REFERENCE_PATH}")
<<<<<<< Updated upstream
=======
    print(f"Logging events to: {EVENT_LOG_PATH}")
    print(f"Temporal smoothing window: {SMOOTHING_WINDOW} frames")
>>>>>>> Stashed changes
    print("Processing only enrolled shelf/item ROI.")
    print("Press 'q' to quit.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read frame.")
            break

        frame_index += 1

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
        match_items_to_slots(detections, slots, frame.shape)
<<<<<<< Updated upstream
        update_stock_levels(slots, histories)
=======
        smooth_slot_counts(slots)          # ← stabilise counts before scoring
        update_stock_levels(slots)
        log_slot_events(slots, previous_slot_states)
        print_count_debug(frame_index, detections, slots)
>>>>>>> Stashed changes

        display_frame = draw_slot_results(frame, slots)

        cv2.imshow("Live Stock Level Test", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()