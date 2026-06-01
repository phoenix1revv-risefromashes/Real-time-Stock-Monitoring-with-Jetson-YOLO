import cv2

from camera import open_camera, read_frame, release_camera

def main():
    camera = open_camera(camera_index=0)
    print("Camera Opened Successfully")



if __name__ == "__main__":
    main()

