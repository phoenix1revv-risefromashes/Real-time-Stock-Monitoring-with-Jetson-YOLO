import cv2

shelf_slots= [
    { 'name': 'A1',
     'x': 0,
     'y': 0,
     'width': 200,
     'height': 200     
     
     }
]



def draw_shelf_slots(frame):
    for slots in shelf_slots:
        x1, y1, x2, y2 = slots['x'], slots['y'], slots['x']+slots['width'], slots['y']+ slots['height']

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),3)

        cv2.putText(frame, slots['name'], (x1, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return frame
    
    