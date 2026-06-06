from datetime import datetime
from pathlib import Path

from cv2 import imwrite

evidence_path = Path("data/evidence/events")

def save_event_evidence(frame, slot_name, previous_status, current_status):
    evidence_path.mkdir(parents=True,exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    filename= f"{timestamp}_{slot_name}_{previous_status}_to_{current_status}.jpg"

    evidence = evidence_path / filename

    imwrite(str(evidence), frame)

    return evidence


