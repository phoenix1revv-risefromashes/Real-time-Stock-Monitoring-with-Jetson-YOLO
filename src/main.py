import cv2
import time

from camera import open_camera, read_frame, release_camera, get_resolution
from detection_logger import log_detection, log_event_detection 
from evidence_capture import save_event_evidence

from shelf_config import *




def main():
    camera = open_camera(camera_index=0)
    
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FPS,30)

    log_interval_time = 2
    last_log_time = 0
    previous_slot_status = {}

    while True:
        frame= read_frame(camera)
        processed_frame, detection_results = draw_shelf_slots_and_identify_occupancy(frame)

        for result in detection_results:
            slot_name = result['slot_name']
            current_status = result['status']
            
            previous_status = previous_slot_status.get(slot_name)

            if previous_status is not None and previous_status != current_status:
                log_event_detection(slot_name, previous_status, current_status, result['edge_pixels'], result['threshold'])
                save_event_evidence(processed_frame,slot_name,previous_status,current_status)

            previous_slot_status[slot_name] = current_status
                

        current_time = time.time()

        if current_time - last_log_time >=log_interval_time:
            for results in detection_results:
                log_detection(results["slot_name"],results['status'], results['edge_pixels'], results['threshold'])
                
               
            last_log_time = current_time




        cv2.imshow(f"Smart Shelf Monitor --Res: {get_resolution(frame)}, FPS: {camera.get(cv2.CAP_PROP_FPS)} ", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    release_camera(camera)
    cv2.destroyAllWindows


    

if __name__ == "__main__":
    main()

