import cv2
import os

def makedir (directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        return None
    else:
        pass

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
 
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
 
    return cv2.resize(image, dim, interpolation=inter)

video_file_path = 'Too much pronation (1).mp4'
cap = cv2.VideoCapture(video_file_path)

i=0
image_count = 0

while i < 3:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    #define region of interest
    roi = frame[500:800, 500:1500]
    roi = ResizeWithAspectRatio(roi, height=200)
    cv2.imshow('roi', roi)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (28, 28), interpolation= cv2.INTER_AREA)

    cv2.imshow('roi scaled and gray', roi)
    copy = frame.copy()
    cv2.rectangle(copy, (500, 500), (800, 1500), (255, 0, 0), 5)

    if i == 0:
        image_count = 0
        cv2.putText(copy, "Hit Enter to Record When Ready", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
    if i == 1:
        image_count += 1
        cv2.putText(copy, "Recording First Gesture - Train", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        cv2.putText(copy, str(image_count), (400, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        gesture_one = './handgestures/train/0/'
        makedir(gesture_one)
        cv2.imwrite(gesture_one + str(image_count) + ".jpg", roi)
    if i == 2:
        image_count += 1
        cv2.putText(copy, "Recording First Gesture - Test", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        cv2.putText(copy, str(image_count), (400, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        gesture_one = './handgestures/test/0/'
        makedir(gesture_one)
        cv2.imwrite(gesture_one + str(image_count) + ".jpg", roi)

    copy = ResizeWithAspectRatio(copy, height=800)
    cv2.imshow('frame', copy)

    if cv2.waitKey(1) == 13:
        image_count = 0
        i += 1

cap.release()