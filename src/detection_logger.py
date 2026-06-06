from datetime import datetime
from pathlib import Path
import csv


log_path = Path('data/logs/detection_log.csv')

event_log_path = Path('data/logs/status_events.csv')

def initialize_log_file(log_path = log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if not log_path.exists():
        with open(log_path, mode='w', newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "slot_name", "status", "edge_pixels", "threshold"])


def initialize_event_log_file(events_path =event_log_path):
    events_path.parent.mkdir(parents=True, exist_ok=True)

    if not events_path.exists():
        with open(events_path, mode='w', exist_ok=True) as file:
            writer = csv.writer(file)
            writer.writerow([
                "timestamp",
                "slot_name",
                "previous_status",
                "current_status",
                "edge_pixels",
                "threshold",
                "event_type"

            ])


def log_event_detection(slot_name, previous_status, current_status, edge_pixels, threshold, events_path=event_log_path ):
    
    initialize_event_log_file()
    
    timestamp = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    with open(event_log_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow[
            timestamp,
            slot_name,
            previous_status,
            current_status,
            edge_pixels,
            threshold,
            "STATUS CHANGED"
        ]

    



def log_detection(slot_name, status, edge_pixels, threshold, log_path=log_path):
    initialize_log_file(log_path=log_path)
   
    timestamp = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    with open (log_path, mode='a', newline="") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, slot_name, status, edge_pixels, threshold])



