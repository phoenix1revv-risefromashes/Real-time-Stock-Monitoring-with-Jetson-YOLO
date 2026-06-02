import cv2

from camera import open_camera, read_frame, release_camera

from shelf_config import SHELF_SLOTS


def main():
    camera = open_camera(camera_index=0)
    print("Camera Opened Successfully\n\n")
    while True:
        frame= read_frame(camera)
        for slots in SHELF_SLOTS:
            x= slots['x']
            y=slots['y']
            width = slots['width']
            height= slots['height']

            top_left = (x,y)
            bottom_right = (x+ width, y + height)

            cv2.rectangle(frame, top_left,bottom_right,(0,255,0), 2)


        #print(f"frame shape: {frame.shape}")

        cv2.imshow("Testing Frames: ", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    release_camera(camera)
    cv2.destroyAllWindows


    






if __name__ == "__main__":
    main()

