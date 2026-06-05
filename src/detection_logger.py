from datetime import datetime
from pathlib import Path
import csv


log_path = Path('data/logs/detection_log.csv')

def initialize_log_file(log_path = log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if not log_path.exists():
        with open(log_path, mode='w', newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "slot_name", "status", "edge_pixels", "threshold"])



def log_detection(slot_name, status, edge_pixels, threshold, log_path=log_path):
    initialize_log_file(log_path=log_path)
   
    timestamp = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    with open (log_path, mode='a', newline="") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, slot_name, status, edge_pixels, threshold])



