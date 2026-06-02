import cv2

from camera import open_camera, read_frame, release_camera

def main():
    camera = open_camera(camera_index=0)
    print("Camera Opened Successfully\n\n")
    while True:
        frame= read_frame(camera)
        #print(f"frame shape: {frame.shape}")

        cv2.imshow("Testing Frames: ", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    release_camera(camera)
    cv2.destroyAllWindows


    






if __name__ == "__main__":
    main()

