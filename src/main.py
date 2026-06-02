import cv2

from camera import open_camera, read_frame, release_camera

from shelf_config import *


def main():
    camera = open_camera(camera_index=0)

    while True:
        frame= read_frame(camera)
        draw_shelf_slots(frame)

        cv2.imshow("Smart Shelf Monitor: ", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    release_camera(camera)
    cv2.destroyAllWindows


    

if __name__ == "__main__":
    main()

