import cv2

from camera import open_camera, read_frame, release_camera, get_resolution

from shelf_config import *


def main():
    camera = open_camera(camera_index=0)

    while True:
        frame= read_frame(camera)
        draw_shelf_slots_and_identify_occupancy(frame)

        cv2.imshow(f"Smart Shelf Monitor --Res: {get_resolution(frame)}, FPS: {camera.get(cv2.CAP_PROP_FPS)} ", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    release_camera(camera)
    cv2.destroyAllWindows


    

if __name__ == "__main__":
    main()

