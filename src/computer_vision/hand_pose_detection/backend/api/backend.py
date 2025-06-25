import base64
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.framework.formats import landmark_pb2

import os
import sys
from pathlib import Path
import supervision as sv
import ultralytics
from ultralytics import YOLO
import torch
from PIL import Image

from datetime import datetime

# import bow code
hand_pose_detection_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(hand_pose_detection_folder_path)
from new_classification_fixed import Classification as bow_class

# Gesture model for hands

# option setup for gesture recognizer
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# gesture model path (set path to gesture_recognizer_custom.task)
hand_pose_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
gesture_model = os.path.join(hand_pose_dir, "gesture_recognizer_custom.task")

#Point 2D class
# A class that stores methods/data for 2d points on the screen
class Point2D:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point2D({self.x}, {self.y})"
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def distance_to(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

    def as_tuple(self):
        return (self.x, self.y)
    
    def to_dict(self):
        return {'x': self.x, 'y': self.y}
    
    def find_point_p1(A, B, ratio=0.7):
        """
        Finds the coordinates of point P1 that is `ratio` distance from A to B.
        
        Parameters:
        A (Point2D): Point A
        B (Point2D): Point B
        ratio (float): Ratio of the distance from A to B where P1 should be (default is 0.7)
        
        Returns:
        Point2D: Coordinates of point P1
        """
        Px = A.x + ratio * (B.x - A.x)
        Py = A.y + ratio * (B.y - A.y)
        return Point2D(Px, Py)

    def find_intersection(p1, p2, p3, p4):
        # Line 1: passing through p1 and p2
        A1 = p2.y - p1.y  # y2 - y1
        B1 = p1.x - p2.x  # x1 - x2
        C1 = A1 * p1.x + B1 * p1.y
 
        # Line 2: passing through p3 and p4
        A2 = p4.y - p3.y  # y4 - y3
        B2 = p3.x - p4.x  # x3 - x4
        C2 = A2 * p3.x + B2 * p3.y
 
        # Determinant of the system
        det = A1 * B2 - A2 * B1
 
        if det == 0:
            # Lines are parallel (no intersection)
            return None
        else:
            # Lines intersect, solving for x and y
            x = (B2 * C1 - B1 * C2) / det
            y = (A1 * C2 - A2 * C1) / det
        return Point2D(x, y)
    
    def is_above_or_below(self, A, B):
        """
        Determines if the current point (self) is above or below the line segment defined by points A and B.
        Parameters:
        A (Point2D): First endpoint of the line segment.
        B (Point2D): Second endpoint of the line segment.
        Returns:
        bool: True if the current point (self) is above the line, False if it is below or on the line.
        """
        # Calculate the cross product of vectors AB and AC (where C is self)
        cross_product = (B.x - A.x) * (self.y - A.y) - (B.y - A.y) * (self.x - A.x)
        if cross_product > 0:
            return True  # Current point (self) is above the line
        else:
            return False 

# Codes for bow verticals
bow_verticals = {
    -3: "Calibrating. Keep bow away from strings",
    -2: "Invalid",
    -1: "Invalid",
    0: "Correct",
    1: "Outside Bow Zone",
    2: "Too Low",
    3: "Too High"
}

def parse_bow(coord_list, classification_list, bow_dict):
    if (bow_dict["bow"] != None):
        coord_list.append(("box bow top left", Point2D(bow_dict["bow"][0][0], bow_dict["bow"][0][1])))
        coord_list.append(("box bow top right", Point2D(bow_dict["bow"][1][0], bow_dict["bow"][1][1])))
        coord_list.append(("box bow bottom left", Point2D(bow_dict["bow"][2][0], bow_dict["bow"][2][1])))
        coord_list.append(("box bow bottom right", Point2D(bow_dict["bow"][3][0], bow_dict["bow"][3][1])))
    if (bow_dict["string"] != None):
        coord_list.append(("box string top left", Point2D(bow_dict["string"][0][0], bow_dict["string"][0][1])))
        coord_list.append(("box string top right", Point2D(bow_dict["string"][1][0], bow_dict["string"][1][1])))
        coord_list.append(("box string bottom left", Point2D(bow_dict["string"][2][0], bow_dict["string"][2][1])))
        coord_list.append(("box string bottom right", Point2D(bow_dict["string"][3][0], bow_dict["string"][3][1])))
    classification_list.append(("bow vertical", bow_verticals[bow_dict["class"]]))

def processFrame(image):
    bow_instance = bow_class()
    bow_dict = bow_instance.process_frame(image)
    
    coord_list = []
    classification_list = []

    parse_bow(coord_list, classification_list, bow_dict)

    return coord_list, classification_list

'''
Adapted the old videoFeed function. It processes each frame and turns it into a video processed with points and classifications.
There's some extra code for the supination cv text that should be adapted once that comes in. 
'''
def videoFeed(video_path_arg, output_path):
    # video capture setup
    cap = cv2.VideoCapture(video_path_arg)
    if not cap.isOpened():
        raise Exception("Failed to open video file")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    output_frame_length = 960
    output_frame_width = 720

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_frame_length, output_frame_width))
    if not out.isOpened():
        raise Exception("Failed to create output video file")
    
    desired_fps = 30 
    frame_delay = int(1000 / desired_fps)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        coord_list, classification_list = processFrame(image)
        frame_count += 1
        """
        if (frame_count % 30 == 0):
                if max(num_correct, num_none, num_supination) == num_supination:
                    display_gesture = "supination"
                elif max(num_correct, num_none, num_supination) == num_correct:
                    display_gesture = "correct"
                else:
                    display_gesture = "none"
                num_none = 0
                num_supination = 0
                num_correct = 0
        """

        #Handling bow box and string box points
        radius = 5           # Radius of the dot
        thickness = -1       # Thickness -1 fills the circle, creating a dot
        if "box bow top left" in coord_list:
            color = (73, 34, 124)
            # SHOWING DOTS
            cv2.circle(image, (int(coord_list["box bow top left"][0]), int(coord_list["box bow top left"][1])), radius, color, thickness)
            cv2.circle(image, (int(coord_list["box bow top right"][0]), int(coord_list["box bow top right"][1])), radius, color, thickness)
            cv2.circle(image, (int(coord_list["box bow bottom left"][0]), int(coord_list["box bow bottom left"][1])), radius, color, thickness)
            cv2.circle(image, (int(coord_list["box bow bottom right"][0]), int(coord_list["box bow bottom right"][1])), radius, color, thickness)
        
        if "box string top left" in coord_list:
            # Define the color and size of the dot
            color = (73, 34, 124)
            # SHOWING DOTS
            cv2.circle(image, (int(coord_list["box string top left"][0]), int(coord_list["box string top left"][1])), radius, color, thickness)
            cv2.circle(image, (int(coord_list["box string top right"][0]), int(coord_list["box string top right"][1])), radius, color, thickness)
            cv2.circle(image, (int(coord_list["box string bottom left"][0]), int(coord_list["box string bottom left"][1])), radius, color, thickness)
            cv2.circle(image, (int(coord_list["box string bottom right"][0]), int(coord_list["box string bottom right"][1])), radius, color, thickness)

        #Putting Text for each frame
        image = cv2.putText(
            image,
            "Frame {}".format(frame_count),
            (10, 50),
            cv2.QT_FONT_NORMAL,
            1,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )
    
        # Resize to specified output dimensions before writing
        resized_frame = cv2.resize(image, (output_frame_length, output_frame_width))

        out.write(resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    #print(Path(__file__).parent.parent)
    path = str(Path(__file__).parent / "demo.avi")
    print(path)
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path
    else:
        raise Exception("Failed to create output video file or file is empty")
  

if __name__ == "__main__":
    print(videoFeed("Cello_backend_test.mp4", "_backend_test.mp4"))

