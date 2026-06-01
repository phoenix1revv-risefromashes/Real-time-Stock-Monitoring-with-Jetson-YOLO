import cv2

def open_camera(camera_index=0):
    camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera index: {camera_index}")
    
    return camera



def read_frame(camera):
    success, frames = camera.read()

    if not success:
        raise RuntimeError("Could not read frames from Camera")
    
    return frames



def release_camera(camera):
    camera.release()
    