import cv2
import yaml




config_path = "configs/shelf_slots.yaml"

def load_shelf_slots (config_path = config_path):
    with open(config_path,'r') as file:
        config = yaml.safe_load(file)

        return config['slots']

shelf_slots = load_shelf_slots()



def draw_shelf_slots_and_identify_occupancy(frame):

    detection_results = []

    for slots in shelf_slots:
        x1, y1, x2, y2 = slots['x'], slots['y'], slots['x']+slots['width'], slots['y']+ slots['height']

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),3)

        occupied, edge_pixels = check_slot_occupancy(frame, slots)

        if occupied:
            status = " Detected"
            color = (0,255,0)
        
        else:
            status = "Empty"
            color = (0,0,255)
        
        detection_results.append({
            'slot_name': slots['name'],
            'status': status,
            'edge_pixels': edge_pixels,
            'threshold': slots['threshold']
        })
            

        cv2.putText(frame, 
                    f"{slots['name'] }: {status}", 
                    (x1+70, y1 + 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4, 
                    color, 
                    2)

    return frame, detection_results
    


def check_slot_occupancy(frame, slots):
    x, y, width, height = slots['x'], slots['y'], slots['x']+slots['width'], slots['y']+ slots['height']

    threshold = slots['threshold']

    roi =frame[y:y+height, x:x+width]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5), 0)
    edges = cv2.Canny(blur,50,150)

    edge_pixels =cv2.countNonZero(edges)
    
    occupied = edge_pixels > threshold


    return occupied , edge_pixels





