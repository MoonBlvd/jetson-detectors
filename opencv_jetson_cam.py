import cv2

video_capture = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)I420, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
while True:
    video_capture_result, frame = video_capture.read()
    if video_capture_result == False:
        raise ValueError('Error reading the frame from camera')
    cv2.imshow('Input', frame)
    if cv2.waitKey(1) == 27:
        break


