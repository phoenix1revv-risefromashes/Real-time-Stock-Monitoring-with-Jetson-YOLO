import cv2



def open_camera(camera_index=0):
    camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera index: {camera_index}")
    
    return camera



def read_frame(camera):
    success, frame = camera.read()

    if not success:
        raise RuntimeError("Could not read frames from Camera")
    
    return frame

def get_resolution(frame):
    height, width = frame.shape[:2]
    return (width, height)
    



def release_camera(camera):
    camera.release()
