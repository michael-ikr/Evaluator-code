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

import ffmpeg

# import bow and hands code
hand_pose_detection_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(hand_pose_detection_folder_path)
from new_classification_fixed import Classification as bow_class
from hands_classification import Hands as hands_class


# option setup for gesture recognizer
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# gesture model path (set path to gesture_recognizer_custom.task)
hand_pose_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
gesture_model = os.path.join(hand_pose_dir, "gesture_recognizer_custom.task")


# Codes for bow verticals
bow_verticals = {
    -3: "Calibrating. Keep bow away from strings",
    -2: "Invalid, no points",
    -1: "Invalid, some points",
    0: "Correct",
    1: "Outside Bow Zone",
    2: "Too Low",
    3: "Too High"
}

bow_angle_types = {
    0: "Correct",
    1: "Incorrect"
}

def parse_bow(coord_list, classification_list, bow_dict):
    if (bow_dict["bow"] != None):
        coord_list.update({"box bow top left": (bow_dict["bow"][0][0], bow_dict["bow"][0][1])})
        coord_list.update({"box bow top right": (bow_dict["bow"][1][0], bow_dict["bow"][1][1])})
        coord_list.update({"box bow bottom left": (bow_dict["bow"][2][0], bow_dict["bow"][2][1])})
        coord_list.update({"box bow bottom right": (bow_dict["bow"][3][0], bow_dict["bow"][3][1])})
    if (bow_dict["string"] != None):
        coord_list.update({"box string top left": (bow_dict["string"][0][0], bow_dict["string"][0][1])})
        coord_list.update({"box string top right": (bow_dict["string"][1][0], bow_dict["string"][1][1])})
        coord_list.update({"box string bottom left": (bow_dict["string"][2][0], bow_dict["string"][2][1])})
        coord_list.update({"box string bottom right": (bow_dict["string"][3][0], bow_dict["string"][3][1])})
    if (bow_dict["class"] != None):
        classification_list.update({"bow vertical": bow_verticals[bow_dict["class"]]})
    if (bow_dict["angle"] != None):
        classification_list.update({"bow angle": bow_angle_types[bow_dict["angle"]]})
        '''
    if (bow_dict["midlines"] != None):
        print("Midlines: ", bow_dict["midlines"])
    if (bow_dict["vertical lines"] != None):
        print("Vertical lines: ", bow_dict["vertical lines"])
        '''

def parse_hands(coord_list, classification_list, hands_dict):
    wrist_posture = hands_dict[0]
    elbow_posture = hands_dict[1]
    hand_coordinates = hands_dict[2]

    classification_list.update({"wrist posture": str(wrist_posture)})
    classification_list.update({"elbow posture": str(elbow_posture)})

    #print("hands coords: ", hand_coordinates)

    coord_list.update({"hand points": hand_coordinates})

    #print("Hands points: ", hand_coordinates)

def processFrame(bow_instance, hands_instance, image):
    #Have the process frame take in a bow class object
    bow_dict = bow_instance.process_frame(image)
    hands_results = hands_instance.process_frame(image)

    coord_list = {}
    classification_list = {}

    parse_bow(coord_list, classification_list, bow_dict)
    parse_hands(coord_list, classification_list, hands_results)

    

    return coord_list, classification_list

# compresses and adds audio
def combineAudio(original_video_path, output_video_path):
    try:
        out_path = "temp49802.mp4"
        input_audio = ffmpeg.input(original_video_path).audio
        new_video = ffmpeg.input(output_video_path).video

        #audio
        ffmpeg.output(new_video, input_audio, out_path, shortest=None, vcodec='copy', acodec='aac').run(overwrite_output=True)
        os.replace(out_path, output_video_path)
    except ffmpeg.Error as e:
        print(e)

def compressVideo(output_video_path):
    try:
        out_path = "temp49802.mp4"
        input_video = ffmpeg.input(output_video_path)

        ffmpeg.output(input_video, out_path, vcodec="libx264", crf=28, preset='medium', acodec='aac', audio_bitrate='128k').run(overwrite_output=True)
        os.replace(out_path, output_video_path)
    except ffmpeg.Error as e:
        print(e)

'''
Adapted the old videoFeed function. It processes each frame and turns it into a video processed with points and classifications.
There's some extra code for the supination cv text that should be adapted once that comes in. 
'''
def videoFeed(video_path_arg, output_path):
    bow_instance = bow_class()
    hands_instance = hands_class()
    # video capture setup
    cap = cv2.VideoCapture(video_path_arg)
    if not cap.isOpened():
        raise Exception("Failed to open video file")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    output_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("resolution", output_frame_width, "x", output_frame_height)
    no_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_finished = 0

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_frame_width, output_frame_height))
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
        coord_list, classification_list = processFrame(bow_instance, hands_instance, image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print("coord list: ", coord_list)
        print("class list: ", classification_list)
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

        #Handling bow box points
        radius = 5           # Radius of the dot
        thickness = -1       # Thickness -1 fills the circle, creating a dot
        text_offset = 35

        if "box bow top left" in coord_list:
            color = (0, 255, 0)
            text_color = (211, 100, 100)
            line_thickness = 2
            # SHOWING DOTS
            box_bow_top_left_pt = (int(coord_list["box bow top left"][0]), int(coord_list["box bow top left"][1]))
            box_bow_top_right_pt = (int(coord_list["box bow top right"][0]), int(coord_list["box bow top right"][1]))
            box_bow_bottom_left_pt = (int(coord_list["box bow bottom left"][0]), int(coord_list["box bow bottom left"][1]))
            box_bow_bottom_right_pt = (int(coord_list["box bow bottom right"][0]), int(coord_list["box bow bottom right"][1]))

            cv2.circle(image, box_bow_top_left_pt, radius, color, thickness)
            cv2.circle(image, box_bow_top_right_pt, radius, color, thickness)
            cv2.circle(image, box_bow_bottom_left_pt, radius, color, thickness)
            cv2.circle(image, box_bow_bottom_right_pt, radius, color, thickness)

            # Prepare text
            text_one = "Bow OBB Coords:"
            text_coord1 = f"Coord 1: ({coord_list['box bow top left'][0]}, {coord_list['box bow top left'][1]})"
            text_coord2 = f"Coord 2: ({coord_list['box bow top right'][0]}, {coord_list['box bow top right'][1]})"
            text_coord3 = f"Coord 3: ({coord_list['box bow bottom left'][0]}, {coord_list['box bow bottom left'][1]})"
            text_coord4 = f"Coord 4: ({coord_list['box bow bottom right'][0]}, {coord_list['box bow bottom right'][1]})"

            #Bounding box lines
            cv2.line(image, box_bow_top_left_pt, box_bow_top_right_pt, color, line_thickness)
            cv2.line(image, box_bow_top_left_pt, box_bow_bottom_left_pt, color, line_thickness)
            cv2.line(image, box_bow_bottom_left_pt, box_bow_bottom_right_pt, color, line_thickness)
            cv2.line(image, box_bow_top_right_pt, box_bow_bottom_right_pt, color, line_thickness)
    
            # Define bottom left corners for each text line
            bottom_left_corner_text_one = (image.shape[1] - 370, 35 * 6 + 20)
            bottom_left_corner_coord1 = (image.shape[1] - 370, 35 * 7 + 15)
            bottom_left_corner_coord2 = (image.shape[1] - 370, 35 * 8 + 10)
            bottom_left_corner_coord3 = (image.shape[1] - 370, 35 * 9 + 5)
            bottom_left_corner_coord4 = (image.shape[1] - 370, 35 * 10 + 0)
            
            # Put text on image for box one
            cv2.putText(image, text_one, bottom_left_corner_text_one, cv2.FONT_HERSHEY_SIMPLEX, .8, text_color, 2)
            cv2.putText(image, text_coord1, bottom_left_corner_coord1, cv2.FONT_HERSHEY_SIMPLEX, .8, text_color, 2)
            cv2.putText(image, text_coord2, bottom_left_corner_coord2, cv2.FONT_HERSHEY_SIMPLEX, .8, text_color, 2)
            cv2.putText(image, text_coord3, bottom_left_corner_coord3, cv2.FONT_HERSHEY_SIMPLEX, .8, text_color, 2)
            cv2.putText(image, text_coord4, bottom_left_corner_coord4, cv2.FONT_HERSHEY_SIMPLEX, .8, text_color, 2)
        
        #String box points handled similarly to the bows
        if "box string top left" in coord_list:
            # Define the color and size of the dot
            color = (0, 255, 0)
            text_color = (0, 255, 0)
            line_thickness = 2
            # SHOWING DOTS
            top_left_pt = (int(coord_list["box string top left"][0]), int(coord_list["box string top left"][1]))
            top_right_pt = (int(coord_list["box string top right"][0]), int(coord_list["box string top right"][1]))
            bottom_left_pt = (int(coord_list["box string bottom left"][0]), int(coord_list["box string bottom left"][1]))
            bottom_right_pt = (int(coord_list["box string bottom right"][0]), int(coord_list["box string bottom right"][1]))

            cv2.circle(image, top_left_pt, radius, color, thickness)
            cv2.circle(image, top_right_pt, radius, color, thickness)
            cv2.circle(image, bottom_left_pt, radius, color, thickness)
            cv2.circle(image, bottom_right_pt, radius, color, thickness)

            #bounding box lines
            cv2.line(image, top_left_pt, top_right_pt, color, line_thickness)
            cv2.line(image, bottom_left_pt, bottom_right_pt, color, line_thickness)
            cv2.line(image, top_left_pt, bottom_right_pt, color, line_thickness)
            cv2.line(image, top_right_pt, bottom_left_pt, color, line_thickness)

            # Prepare text
            text_one = "String OBB Coords:"
            text_coord1 = f"Coord 1: ({coord_list['box string top left'][0]}, {coord_list['box string top left'][1]})"
            text_coord2 = f"Coord 2: ({coord_list['box string top right'][0]}, {coord_list['box string top right'][1]})"
            text_coord3 = f"Coord 3: ({coord_list['box string bottom left'][0]}, {coord_list['box string bottom left'][1]})"
            text_coord4 = f"Coord 4: ({coord_list['box string bottom right'][0]}, {coord_list['box string bottom right'][1]})"
    
            text_offset = 35  # increased spacing between lines
            top_right_corner_text_two = (image.shape[1] - 370, text_offset + 20) # Adjusted to move down and left
            top_right_corner_coord1_2 = (image.shape[1] - 370, text_offset * 2 + 15) # Adjusted to move down and left
            top_right_corner_coord2_2 = (image.shape[1] - 370, text_offset * 3 + 10) # Adjusted to move down and left
            top_right_corner_coord3_2 = (image.shape[1] - 370, text_offset * 4 + 5) # Adjusted to move down and left
            top_right_corner_coord4_2 = (image.shape[1] - 370, text_offset * 5 + 0) # Adjusted to move down and left
            
            # Put text on image for box one
            cv2.putText(image, text_one, top_right_corner_text_two, cv2.FONT_HERSHEY_SIMPLEX, .8, text_color, 2)
            cv2.putText(image, text_coord1, top_right_corner_coord1_2, cv2.FONT_HERSHEY_SIMPLEX, .8, text_color, 2)
            cv2.putText(image, text_coord2, top_right_corner_coord2_2, cv2.FONT_HERSHEY_SIMPLEX, .8, text_color, 2)
            cv2.putText(image, text_coord3, top_right_corner_coord3_2, cv2.FONT_HERSHEY_SIMPLEX, .8, text_color, 2)
            cv2.putText(image, text_coord4, top_right_corner_coord4_2, cv2.FONT_HERSHEY_SIMPLEX, .8, text_color, 2)
        
        #Hands points
        if "hand points" in coord_list:
            color = (0, 255, 0)
            hands_points = coord_list["hand points"]
            for point in hands_points:
                if (not isinstance(point, tuple)):
                    continue
                if (len(point) < 2):
                    break
                hand_x = point[0] * output_frame_width
                hand_y = point[1] * output_frame_height
                point_tuple = (int(hand_x), int(hand_y))
                cv2.circle(image, point_tuple, 5, color, thickness)
                #print("x: ", int(hand_x), " y: ", int(hand_y))

        #Handling bow vertical classification video text
        if "bow vertical" in classification_list:
            text_color = (255, 0, 0)
            bow_text_coord = (image.shape[1] - 370, text_offset * 11 + 0) # Adjusted to move down and left
            cv2.putText(image, ("Bow: " + classification_list["bow vertical"]), bow_text_coord, cv2.FONT_HERSHEY_SIMPLEX, .8, text_color, 4)

        #Handling bow angle classification video text
        if "bow angle" in classification_list:
            text_color = (255, 0, 0)
            bow_text_coord = (image.shape[1] - 370, text_offset * 13 + 0) # Adjusted to move down and left
            cv2.putText(image, ("Bow angle: " + classification_list["bow angle"]), bow_text_coord, cv2.FONT_HERSHEY_SIMPLEX, .8, text_color, 4)


        #Handling wrist posture video text
        if "wrist posture" in classification_list:
            text_color = (255, 0, 0)
            wrist_text_coord = (image.shape[1] - 370, text_offset * 15 + 0)
            cv2.putText(image, ("Wrist: " + classification_list["wrist posture"]), wrist_text_coord, cv2.FONT_HERSHEY_SIMPLEX, .8, text_color, 4)
        
        
        #Handling elbow posture video text
        if "elbow posture" in classification_list:
            text_color = (255, 0, 0)
            elbow_text_coord = (image.shape[1] - 370, text_offset * 17 + 0)
            cv2.putText(image, ("Elbow: " + classification_list["elbow posture"]), elbow_text_coord, cv2.FONT_HERSHEY_SIMPLEX, .8, text_color, 4)

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

        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #image_height, image_width, _ = image.shape
    
        # Resize to specified output dimensions before writing

        out.write(image)

        frames_finished += 1
        print(frames_finished, "/", no_frames, " frames processed")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    #print(Path(__file__).parent.parent)
    path = str(Path(__file__).parent / "demo.avi")
    print(path)
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        combineAudio(video_path_arg, output_path)
        compressVideo(output_path)
        return output_path
    else:
        raise Exception("Failed to create output video file or file is empty")
    
    
  

if __name__ == "__main__":
    #print(videoFeed("cello_backend_test_v3.mp4", "_backend_test.mp4"))
    #print(ffmpegPostProcessVideo("cello_backend_test_v3.mp4", "_backend_test_v2.mp4"))

    #print(videoFeed("C:/Users/ljcho/Downloads/supination and bow high - slow.mp4", "_backend_test_v4.mp4"))
    #print(ffmpegPostProcessVideo("C:/Users/ljcho/Downloads/supination and bow high - slow.mp4", "_backend_test_v4.mp4"))

    

    #print(videoFeed("C:/Users/ljcho/Downloads/bow too high-slow.mp4", "_backend_test_v5.mp4"))
    #print(combineAudio("C:/Users/ljcho/Downloads/bow too high-slow.mp4", "_backend_test_v5.mp4"))

    compressVideo("_backend_test_v5.mp4")