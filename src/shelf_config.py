import cv2



shelf_slots= [
    { 'name': 'A1',
     'x': 0,
     'y': 0,
     'width': 200,
     'height': 350     
     
     },

     {
         'name': 'A2',
         'x': 220,
         'y' : 0,
         'width': 415,
         'height' : 140
     },

     {
         'name': 'A3',
         'x' : 220,
         'y' : 145,
         'width': 415,
         'height': 210
     }
]



def draw_shelf_slots_and_identify_occupancy(frame):
    for slots in shelf_slots:
        x1, y1, x2, y2 = slots['x'], slots['y'], slots['x']+slots['width'], slots['y']+ slots['height']

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),3)

        occupied = check_slot_occupancy(frame, slots)

        if occupied:
            status = " Detected"
            color = (0,255,0)
        
        else:
            status = "Empty"
            color = (0,0,255)
            

        cv2.putText(frame, 
                    f"{slots['name'] }: {status}", 
                    (x1+70, y1 + 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4, 
                    color, 
                    2)

    return frame
    


def check_slot_occupancy(frame, slots):
    x, y, width, height = slots['x'], slots['y'], slots['x']+slots['width'], slots['y']+ slots['height']

    roi =frame[y:y+height, x:x+width]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5), 0)
    edges = cv2.Canny(blur,50,150)

    edge_pixels =cv2.countNonZero(edges)
    

    return edge_pixels>1500



