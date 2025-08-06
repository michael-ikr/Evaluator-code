import cv2
import mediapipe as mp
import numpy as np
import csv

from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.framework.formats import landmark_pb2

import copy
import itertools

import os
import sys
import supervision as sv
import ultralytics
from ultralytics import YOLO
import torch

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from datetime import datetime

base_directory = os.path.dirname(__file__)

# Calculate the correct path to the model directory
model_directory = os.path.join(base_directory, 'model', 'keypoint_classifier')
model_file = os.path.join(model_directory, 'keypoint_classifier_v1.tflite')

# Ensure the model file exists
if not os.path.exists(model_file):
    raise FileNotFoundError(f"Model file not found: {model_file}")

# Add the model_directory to sys.path
if model_directory not in sys.path:
    sys.path.append(model_directory)

from model import KeyPointClassifier

# Calculate the path to the 'computer_vision/utils' directory
utils_directory = os.path.join(base_directory, 'computer_vision', 'utils')

# Add the utils_directory to sys.path if it's not already included
if utils_directory not in sys.path:
    sys.path.append(utils_directory)

from utils import CvFpsCalc


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
    
    @staticmethod
    def angle_between_lines(A, B, C, D):
        """
        Calculates the angle between the line segment AB and CD.        
        """
        # Vectors AB and CD
        vector_ab = (B.x - A.x, B.y - A.y)
        vector_cd = (D.x - C.x, D.y - C.y)

        dot_product = vector_ab[0] * vector_cd[0] + vector_ab[1] * vector_cd[1]

        magnitude_ab = A.distance_to(B)
        magnitude_cd = C.distance_to(D)
      
       # Dot product property
        cos_theta = dot_product / (magnitude_ab * magnitude_cd)

        # Calculate arc cosine
        angle_radians = math.acos(cos_theta)
        angle_degrees = math.degrees(angle_radians)
        
        return angle_degrees

# A class for hands proccessing
class Hands:
    # Function to resize image with aspect ratio
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

    # Function to store finger node coordinates
    def store_finger_node_coords(id: int, cx: float, cy: float, finger_coords: dict):
        if id not in finger_coords:
            finger_coords[id] = []
        finger_coords[id].append((cx, cy))

    def calc_bounding_rect(image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv2.boundingRect(landmark_array)

        return [x, y, x + w, y + h]

    def calc_landmark_list(image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # Keypoint
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            # landmark_z = landmark.z

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    def pre_process_landmark(landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # Convert to a one-dimensional list
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list

    def draw_bounding_rect(use_brect, image, brect):
        if use_brect:
            # Outer rectangle
            cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                        (0, 0, 0), 1)

        return image

    def draw_info_text(image, brect, handedness, hand_sign_text):
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                    (0, 0, 0), -1)

        info_text = handedness.classification[0].label[0:]
        #
        if hand_sign_text != "":
            info_text = info_text + ':' + hand_sign_text
        cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        return image

    def draw_info(image, fps, mode, number):
        cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)

        mode_string = ['Logging Key Point', 'Logging Point History']
        if 1 <= mode <= 2:
            cv2.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    cv2.LINE_AA)
            if 0 <= number <= 9:
                cv2.putText(image, "NUM:" + str(number), (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                        cv2.LINE_AA)
        return image

    def draw_landmarks(image, landmark_point):
        if len(landmark_point) > 0:
            # Thumb
            cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                    (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                    (255, 255, 255), 2)

            # Index finger
            cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                    (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                    (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                    (255, 255, 255), 2)

            # Middle finger
            cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                    (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                    (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                    (255, 255, 255), 2)

            # Ring finger
            cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                    (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                    (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                    (255, 255, 255), 2)

            # Little finger
            cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                    (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                    (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                    (255, 255, 255), 2)

            # Palm
            cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                    (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                    (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                    (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                    (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                    (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                    (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                    (255, 255, 255), 2)

        # Key Points
        for index, landmark in enumerate(landmark_point):
            if index == 0:  
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 1: 
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 2: 
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 3:  
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 4:  
                cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                        -1)
                cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 5: 
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 6: 
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 7:  
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 8:  
                cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                        -1)
                cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 9:  
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 10:  
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 11:  
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 12: 
                cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                        -1)
                cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 13: 
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 14:  
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 15: 
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 16:  
                cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                        -1)
                cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 17:  
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 18:  
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 19:  
                cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 20:  
                cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                        -1)
                cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

        return image

    def blur_edge(img, d=31):
        """Blur edges to reduce artifacts in Wiener deconvolution."""
        h, w = img.shape[:2]
        kernel_x = cv2.getGaussianKernel(w, d)
        kernel_y = cv2.getGaussianKernel(h, d)
        kernel = kernel_y @ kernel_x.T  # Outer product to create 2D kernel
        kernel = kernel / kernel.max()  # Normalize
        
        # Expand to 3 channels
        kernel = np.repeat(kernel[:, :, np.newaxis], 3, axis=2)  

        img_blur = cv2.GaussianBlur(img, (d, d), 0)
        return img * kernel + img_blur * (1 - kernel)


    def motion_kernel(angle, d, sz=65):
        kern = np.ones((1, d), np.float32)
        c, s = np.cos(angle), np.sin(angle)
        A = np.float32([[c, -s, 0], [s, c, 0]])
        sz2 = sz // 2
        A[:,2] = (sz2, sz2) - np.dot(A[:,:2], ((d-1)*0.5, 0))
        kern = cv2.warpAffine(kern, A, (sz, sz), flags=cv2.INTER_CUBIC)
        return kern

    def defocus_kernel(d, sz=65):
        kern = np.zeros((sz, sz), np.uint8)
        cv2.circle(kern, (sz, sz), d, 255, -1, cv2.LINE_AA, shift=1)
        kern = np.float32(kern) / 255.0
        return kern

    def wiener_deconvolution(img, angle, snr=0.01):
        """Performs Wiener deconvolution to reduce motion blur."""
        
        # Ensure the image is grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Convert image to float32
        img = np.float32(img)
        
        # Get image size
        h, w = img.shape

        # Generate a motion blur kernel
        def motion_blur_kernel(size, angle):
            """Creates a motion blur kernel with a given size and angle."""
            kernel = np.zeros((size, size), dtype=np.float32)
            center = size // 2
            cv2.line(kernel, (center - size//2, center), (center + size//2, center), 1, thickness=1)
            M = cv2.getRotationMatrix2D((center, center), angle, 1)
            kernel = cv2.warpAffine(kernel, M, (size, size))
            kernel /= np.sum(kernel)  # Normalize
            return kernel

        # Define kernel size (must be odd)
        kernel_size = min(h, w) // 20  # Adjust as needed
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1  # Ensure odd size

        # Create motion blur kernel
        psf = motion_blur_kernel(kernel_size, angle)

        # Convert kernel to frequency domain (DFT)
        psf_padded = np.zeros_like(img, dtype=np.float32)
        kh, kw = psf.shape
        psf_padded[:kh, :kw] = psf
        psf_dft = cv2.dft(psf_padded, flags=cv2.DFT_COMPLEX_OUTPUT)

        # Apply DFT to the image
        img_dft = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)

        # Compute Wiener filter in the frequency domain
        psf_abs2 = psf_dft[:, :, 0]**2 + psf_dft[:, :, 1]**2
        wiener_filter = np.zeros_like(psf_dft)
        wiener_filter[:, :, 0] = psf_dft[:, :, 0] / (psf_abs2 + snr)
        wiener_filter[:, :, 1] = psf_dft[:, :, 1] / (psf_abs2 + snr)

        # Apply the Wiener filter
        deblurred_dft = np.zeros_like(img_dft)
        deblurred_dft[:, :, 0] = img_dft[:, :, 0] * wiener_filter[:, :, 0] + img_dft[:, :, 1] * wiener_filter[:, :, 1]
        deblurred_dft[:, :, 1] = img_dft[:, :, 1] * wiener_filter[:, :, 0] - img_dft[:, :, 0] * wiener_filter[:, :, 1]

        # Convert back to spatial domain
        deblurred_img = cv2.idft(deblurred_dft, flags=cv2.DFT_REAL_OUTPUT)
        
        # Normalize the result to 0-255
        deblurred_img = cv2.normalize(deblurred_img, None, 0, 255, cv2.NORM_MINMAX)
        deblurred_img = np.uint8(deblurred_img)

        return deblurred_img
    
    def apply_wiener_deconvolution(image, angle, apply_deconvolution=True):
        """
        Applies Wiener deconvolution to the given image if apply_deconvolution is True.
        
        Parameters:
        image (numpy.ndarray): The input image (BGR format).
        angle (float): The angle of motion blur.
        apply_deconvolution (bool): Whether to apply the Wiener deconvolution.
        
        Returns:
        numpy.ndarray: The processed image (either deblurred or original).
        """
        if not apply_deconvolution:
            return image  # Return the original image if deconvolution is disabled
        
        # Convert image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Perform Wiener deconvolution
        deblurred_image = Hands.wiener_deconvolution(gray_image, angle)
        
        # Convert back to BGR for display
        return cv2.cvtColor(deblurred_image, cv2.COLOR_GRAY2BGR)
    
    def display_comparison(original, deblurred, show_comparison=True, window_name=None):
        """
        Displays a side-by-side comparison of the original and deblurred images if show_comparison is True.
        
        Parameters:
        original (numpy.ndarray): The original image region.
        deblurred (numpy.ndarray): The deblurred image region.
        show_comparison (bool): Whether to show the comparison window.
        window_name (str, optional): The name of the display window.
        """
        if show_comparison and original.shape == deblurred.shape:
            if window_name is None:
                window_name = 'Original vs Deblurred Hand'
            comparison = cv2.hconcat([original, deblurred])
            cv2.namedWindow(str(window_name), cv2.WINDOW_NORMAL)
            cv2.resizeWindow(str(window_name), comparison.shape[1], comparison.shape[0])
            cv2.imshow(str(window_name), comparison)

    def process_hand_region(debug_image, hand_region, deblur_angle, x1, y1, x2, y2, apply_deconvolution=True, show_comparison=True):
        
        # Add more kernels (or comment out) for testing
        processed_hand = Hands.apply_wiener_deconvolution(hand_region, deblur_angle, apply_deconvolution)
        
        Hands.display_comparison(hand_region, processed_hand, show_comparison)
        debug_image[y1:y2, x1:x2] = processed_hand

    def logging_csv(number, mode, landmark_list, handedness, frame_num):
        if mode == 0:
            pass
        if mode == 1 and (0 <= number <= 9):
            if handedness.classification[0].label[0:] == "Right":
                csv_path = 'src/computer_vision/hand_pose_detection/model/keypoint_classifier/keypoint.csv'
                with open(csv_path, 'a', newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([number, *landmark_list])
                    print("Logged data on frame " + str(frame_num))
        return

    def select_mode(key, mode):
        number = -1
        if 48 <= key <= 57:  # 0 ~ 9
            number = key - 48
        if key == 110:  # n
            mode = 0
        if key == 107:  # k
            mode = 1
        if key == 104:  # h
            mode = 2
        return number, mode

    def evaluate_performance(classification_results, save_path="gesture_classification_results.csv"):
        """
        Evaluates performance metrics based on classification results.
        
        Parameters:
        classification_results (list of dict): List containing classification results with 'ground_truth' and 'predicted' labels.
        save_path (str): Path to save the CSV file.
        
        Returns:
        None
        """
        # Save results to CSV for offline analysis
        df = pd.DataFrame(classification_results)
        df.to_csv(save_path, index=False)
        
        # Filter out frames without ground truth labels
        df = df.dropna()
        
        if df.empty:
            print("No ground truth labels available. Cannot compute accuracy metrics.")
            return
        
        # Compute Performance Metrics
        accuracy = accuracy_score(df["ground_truth"], df["predicted"])
        precision = precision_score(df["ground_truth"], df["predicted"], average='weighted', zero_division=0)
        recall = recall_score(df["ground_truth"], df["predicted"], average='weighted', zero_division=0)
        f1 = f1_score(df["ground_truth"], df["predicted"], average='weighted', zero_division=0)
        
        print(f"Classification Accuracy: {accuracy:.2%}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall: {recall:.2%}")
        print(f"F1-score: {f1:.2%}")

        # Rolling Accuracy Over Time
        rolling_accuracy = df.groupby(df.index // 20).apply(lambda x: accuracy_score(x["ground_truth"], x["predicted"]))
        plt.plot(rolling_accuracy.index, rolling_accuracy.values, marker='o', linestyle='-')
        plt.xlabel("Frame Chunk (20 Frames Each)")
        plt.ylabel("Rolling Accuracy")
        plt.title("Gesture Classification Accuracy Over Time")
        plt.show()
        
        # Generate Confusion Matrix
        labels = sorted(df["ground_truth"].unique())
        cm = confusion_matrix(df["ground_truth"], df["predicted"], labels=labels)
        
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()



def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode

def blur_edge(img, d=31):
    h, w  = img.shape[:2]
    img_pad = cv2.copyMakeBorder(img, d, d, d, d, cv2.BORDER_WRAP)
    img_blur = cv2.GaussianBlur(img_pad, (2*d+1, 2*d+1), -1)[d:-d,d:-d]
    y, x = np.indices((h, w))
    dist = np.dstack([x, w-x-1, y, h-y-1]).min(-1)
    w = np.minimum(np.float32(dist)/d, 1.0)
    return img*w + img_blur*(1-w)

def motion_kernel(angle, d, sz=65):
    kern = np.ones((1, d), np.float32)
    c, s = np.cos(angle), np.sin(angle)
    A = np.float32([[c, -s, 0], [s, c, 0]])
    sz2 = sz // 2
    A[:,2] = (sz2, sz2) - np.dot(A[:,:2], ((d-1)*0.5, 0))
    kern = cv2.warpAffine(kern, A, (sz, sz), flags=cv2.INTER_CUBIC)
    return kern

def defocus_kernel(d, sz=65):
    kern = np.zeros((sz, sz), np.uint8)
    cv2.circle(kern, (sz, sz), d, 255, -1, cv2.LINE_AA, shift=1)
    kern = np.float32(kern) / 255.0
    return kern

def classify_elbow_posture(shoulder, elbow, hand, reference_ratio, threshold=0.1):
    #calculate distances
    shoulder_to_elbow = np.sqrt((shoulder.x - elbow.x)**2 + (shoulder.y - elbow.y)**2)
    elbow_to_hand = np.sqrt((elbow.x - hand.x)**2 + (elbow.y - hand.y)**2)

    # calculate ratio
    distance_ratio = shoulder_to_elbow / elbow_to_hand

    # compare with reference
    if abs(distance_ratio - reference_ratio) > threshold:
        return "Elbow Too High"
    return "Correct Posture"

def main():
    # YOLOv8 model trained from Roboflow dataset
    # Used for bow and target area oriented bounding boxes
    model = YOLO('bow_target.pt')  # Path to your model file

    # Initialization of deconvolution parameters
    defocus = False  # Default to motion kernel, change as needed
    angle = 180  # Default angle for motion blur
    d = 22  # Default diameter
    snr = 25  # Default SNR value

    target_x = 0.5
    target_y = 0.5
    proximity_threshold = 0.1

    # Input video file
    video_file_path = 'Supination.mp4'
    cap = cv2.VideoCapture(video_file_path) # change argument to 0 for demo/camera input

    frame_count = 0
    output_frame_length = 960
    output_frame_width = 720

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # Setup gesture options
    num_hands = 2
    finger_coords = {}

    # Variables for landmark validation and temporal correction
    recent_classifications = []  # Store recent valid classifications
    classification_buffer_size = 10
    
    # Variables for pose landmark validation and temporal correction
    recent_postures = []  # Store recent valid posture classifications
    posture_buffer_size = 10

    # Instantiate gesture classifier
    keypoint_classifier = KeyPointClassifier(model_path=model_file)

    relative_csv_path = 'model/keypoint_classifier/keypoint_classifier_label.csv'
    csv_path = os.path.join(base_directory, relative_csv_path)

    with open(csv_path, encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

    # FPS Measurement
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    mode = 0

    #helper functions I created
    def detect_bad_landmarks(landmark_list):
        if not landmark_list or len(landmark_list) < 21:
            return True
        try:
            x_coords = [point[0] for point in landmark_list]
            y_coords = [point[1] for point in landmark_list]
            x_spread = max(x_coords) - min(x_coords)
            y_spread = max(y_coords) - min(y_coords)
            min_expected_spread = 40
            if x_spread < min_expected_spread or y_spread < min_expected_spread:
                return True
            fingertips = [landmark_list[4], landmark_list[8], landmark_list[12], landmark_list[16], landmark_list[20]]
            close_fingertip_count = 0
            for i in range(len(fingertips)):
                for j in range(i + 1, len(fingertips)):
                    tip1, tip2 = fingertips[i], fingertips[j]
                    distance = np.sqrt((tip1[0] - tip2[0])**2 + (tip1[1] - tip2[1])**2)
                    if distance < 15:
                        close_fingertip_count += 1
            if close_fingertip_count > 2:
                return True
            knuckles = [landmark_list[5], landmark_list[9], landmark_list[13], landmark_list[17]]
            knuckle_x_coords = [point[0] for point in knuckles]
            knuckle_y_coords = [point[1] for point in knuckles]
            knuckle_x_spread = max(knuckle_x_coords) - min(knuckle_x_coords)
            knuckle_y_spread = max(knuckle_y_coords) - min(knuckle_y_coords)
            if knuckle_x_spread < 25 or knuckle_y_spread < 15:
                return True
            return False
        except (IndexError, TypeError, ValueError):
            return True

    def get_recent_valid_classification():
        if not recent_classifications:
            return "Supination"
        classification_counts = {}
        for classification in recent_classifications[-5:]:
            classification_counts[classification] = classification_counts.get(classification, 0) + 1
        if "Supination" in classification_counts:
            return "Supination"
        return max(classification_counts, key=classification_counts.get)

    def detect_bad_pose_landmarks(shoulder, elbow, wrist):
        try:
            upper_arm_length = np.sqrt((elbow.x - shoulder.x)**2 + (elbow.y - shoulder.y)**2)
            forearm_length = np.sqrt((wrist.x - elbow.x)**2 + (wrist.y - elbow.y)**2)
            total_arm_length = np.sqrt((wrist.x - shoulder.x)**2 + (wrist.y - shoulder.y)**2)
            min_upper_arm, min_forearm = 0.05, 0.04
            if upper_arm_length < min_upper_arm or forearm_length < min_forearm:
                return True
            segment_sum = upper_arm_length + forearm_length
            geometry_ratio = total_arm_length / segment_sum if segment_sum > 0 else 0
            if geometry_ratio > 1.2:
                return True
            landmarks_spread_x = max(shoulder.x, elbow.x, wrist.x) - min(shoulder.x, elbow.x, wrist.x)
            landmarks_spread_y = max(shoulder.y, elbow.y, wrist.y) - min(shoulder.y, elbow.y, wrist.y)
            min_expected_spread = 0.08
            if landmarks_spread_x < min_expected_spread and landmarks_spread_y < min_expected_spread:
                return True
            margin = 0.05
            for point in [shoulder, elbow, wrist]:
                if (point.x < margin or point.x > 1.0 - margin or point.y < margin or point.y > 1.0 - margin):
                    return True
            return False
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            return True

    def get_recent_valid_posture():
        if not recent_postures:
            return "Normal"
        posture_counts = {}
        for posture in recent_postures[-5:]:
            posture_counts[posture] = posture_counts.get(posture, 0) + 1
        return max(posture_counts, key=posture_counts.get)

    hands = mp_hands.Hands(
        model_complexity=0,
        max_num_hands=num_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    pose = mp_pose.Pose(
        model_complexity=0,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.6)

    # Store results across frames
    classification_results = []
    
    # Initialize video writer
    writer = cv2.VideoWriter("demo.avi", cv2.VideoWriter_fourcc(*"MJPG"), 12.5,(output_frame_length,output_frame_width))

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
            
        # Process Key (ESC: end)
        key = cv2.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Calculate FPS
        fps = cvFpsCalc.get()

        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        image = cv2.flip(image, 1)
        debug_image = np.copy(image)
        debug_image.flags.writeable = True
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # mp hand and pose models process the frame
        results = hands.process(image)
        pose_results = pose.process(image)
        
        frame_count += 1
        image_height, image_width, _ = image.shape
        
        # Draw hand landmarks
        if results.multi_hand_landmarks and pose_results.pose_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                for ids, landmrk in enumerate(hand_landmarks.landmark):
                    cx, cy = landmrk.x * image_width, landmrk.y * image_height
                    Hands.store_finger_node_coords(ids, cx, cy, finger_coords)
                
                # Check if the hand is the bow hand
                if handedness.classification[0].label == 'Right':
                    brect = Hands.calc_bounding_rect(debug_image, hand_landmarks)
                    landmark_list = Hands.calc_landmark_list(image, hand_landmarks)
                    landmarks_are_bad = detect_bad_landmarks(landmark_list)
                    pre_processed_landmark_list = Hands.pre_process_landmark(landmark_list)
                    Hands.logging_csv(number, mode, pre_processed_landmark_list, handedness, frame_count)

                    # Hand sign classification
                    if landmarks_are_bad:
                        predicted_label = get_recent_valid_classification()
                        print(f"Frame {frame_count}: Bad landmarks detected, using recent classification: {predicted_label}")
                        cv2.putText(debug_image, f"Bad Landmarks - Using Recent: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                        predicted_label = keypoint_classifier_labels[hand_sign_id]
                        recent_classifications.append(predicted_label)
                        if len(recent_classifications) > classification_buffer_size:
                            recent_classifications.pop(0)

                    ground_truth_label = "Normal"
                    classification_results.append({
                        "frame": frame_count, "predicted": predicted_label, "ground_truth": ground_truth_label,
                        "landmarks_bad": landmarks_are_bad, "pose_landmarks_bad": locals().get('pose_landmarks_are_bad', False),
                        "posture": locals().get('posture', 'Not detected')
                    })
                    print("Frame:", frame_count, "predicted:", predicted_label, "ground truth:", ground_truth_label, "bad landmarks:", landmarks_are_bad, "bad pose:", locals().get('pose_landmarks_are_bad', False))

                    # Draw hands and info
                    debug_image = Hands.draw_bounding_rect(True, debug_image, brect)
                    mp_drawing.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
                    debug_image = Hands.draw_info_text(debug_image, brect, handedness, predicted_label)
                    
                    # Get pose landmarks for elbow posture
                    pose_landmarks = pose_results.pose_landmarks.landmark
                    shoulder, elbow, hand = pose_landmarks[12], pose_landmarks[14], pose_landmarks[16]
                    
                    hand_distance = np.sqrt((hand.x - target_x)**2 + (hand.y - target_y)**2)
                    if hand_distance <= proximity_threshold:
                        pose_landmarks_are_bad = detect_bad_pose_landmarks(shoulder, elbow, hand)
                        reference_ratio = 1.2
                        if pose_landmarks_are_bad:
                            posture = get_recent_valid_posture()
                            print(f"Frame {frame_count}: Bad pose landmarks detected, using recent posture: {posture}")
                            cv2.putText(debug_image, f"Bad Pose - Using Recent: {posture}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                        else:
                            posture = classify_elbow_posture(shoulder, elbow, hand, reference_ratio, threshold=0.1)
                            recent_postures.append(posture)
                            if len(recent_postures) > posture_buffer_size:
                                recent_postures.pop(0)

                        text_position = (debug_image.shape[1] - 300, debug_image.shape[0] - 20)
                        cv2.putText(debug_image, f"Posture: {posture}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                        
                        pose_quality_text = "Pose: Valid" if not pose_landmarks_are_bad else "Pose: Invalid"
                        pose_quality_color = (0, 255, 0) if not pose_landmarks_are_bad else (255, 165, 0)
                        cv2.putText(debug_image, pose_quality_text, (debug_image.shape[1] - 300, debug_image.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pose_quality_color, 2)

                # Draw pose subset
                if pose_results.pose_landmarks:
                    landmark_subset = landmark_pb2.NormalizedLandmarkList(landmark=pose_results.pose_landmarks.landmark[11:15])
                    mp_drawing.draw_landmarks(debug_image, landmark_subset, None, mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=10, circle_radius=6))
                    mp_drawing.draw_landmarks(debug_image, landmark_subset, None, mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=10))

        # --- Final Drawing and Display ---
        debug_image = Hands.ResizeWithAspectRatio(debug_image, height=800)
        debug_image = cv2.putText(debug_image, "Frame {}".format(frame_count), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(debug_image, "Frame {}".format(frame_count), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        resized_frame = cv2.resize(debug_image, (output_frame_length, output_frame_width))
        debug_image = Hands.draw_info(debug_image, fps, mode, number)
        writer.write(resized_frame)
        cv2.imshow('MediaPipe Hands', debug_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    hands.close()
    pose.close()
    
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    
    # --- Final Performance Evaluation ---
    if classification_results:
        Hands.evaluate_performance(classification_results)
        total_frames = len(classification_results)
        bad_landmark_frames = sum(1 for result in classification_results if result.get('landmarks_bad', False))
        bad_pose_frames = sum(1 for result in classification_results if result.get('pose_landmarks_bad', False))
        
        print(f"\nLandmark Quality Analysis:")
        print(f"Total frames: {total_frames}")
        print(f"Frames with bad hand landmarks: {bad_landmark_frames}")
        print(f"Frames with bad pose landmarks: {bad_pose_frames}")
        print(f"Hand landmark quality: {((total_frames - bad_landmark_frames) / total_frames * 100):.1f}%")
        print(f"Pose landmark quality: {((total_frames - bad_pose_frames) / total_frames * 100):.1f}%")
        
        bad_hand_frames = [result['frame'] for result in classification_results if result.get('landmarks_bad', False)]
        bad_pose_frames_list = [result['frame'] for result in classification_results if result.get('pose_landmarks_bad', False)]
        
        if bad_hand_frames:
            print(f"Frames with bad hand landmarks: {bad_hand_frames}")
        if bad_pose_frames_list:
            print(f"Frames with bad pose landmarks: {bad_pose_frames_list}")
            
        postures = [result.get('posture', 'Not detected') for result in classification_results if result.get('posture') != 'Not detected']
        if postures:
            posture_counts = {}
            for p in postures:
                posture_counts[p] = posture_counts.get(p, 0) + 1
            print(f"\nPosture Analysis:")
            for p, count in posture_counts.items():
                print(f"{p}: {count} frames ({count/len(postures)*100:.1f}%)")
    else:
        print("No classification results available.")

if __name__ == "__main__":
    main()
