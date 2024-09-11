'''This module should probably have a decent docstring'''

# common imports
import math
import numpy as np



# this is supposedly a fix for long VideoCapture open times on windows with Logi webcams.
# Didn't work for me though
# import os
# os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2

# https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
# import mediapipe
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

from mediapipe import solutions
# pylint seems to REALLY hate this for some reason
from mediapipe.framework.formats import landmark_pb2


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

# TODO
# should the math and image annotation be 2 separate functions?

# TODO
# this should potentially return an array
# if the array is empty, no hands are present, etc.

def calculate_distance(detection_result):
    """IDK DOCSTRING OR WHATEVER"""
    hand_landmarks_list = detection_result.hand_landmarks
    # i honestly do not care about handedness
    # handedness_list = detection_result.handedness

    dist_sum = 0
    # Loop through the detected HANDS to visualize.
    # therefore, this should run twice AT MOST
    for _, hand_landmarks in enumerate(hand_landmarks_list):

        base = hand_landmarks[0]
        fingertips = [
            hand_landmarks[4],
            hand_landmarks[8],
            hand_landmarks[12],
            hand_landmarks[16],
            hand_landmarks[20],
        ]

        dist_sum = 0
        for finger in fingertips:
            dist = math.sqrt(
                pow(finger.x - base.x, 2)
                + pow(finger.y - base.y, 2)
                + pow(finger.z - base.z, 2)
            )
            dist_sum += dist
    return dist_sum / 5


# very nice function!
# source:
# https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#scrollTo=_JVO3rvPD4RN&uniqifier=1
def annotate_image(rgb_image, detection_result, text):
    ''' takes an image as input and draws hands as well as values '''
    hand_landmarks_list = detection_result.hand_landmarks
    # handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)
    height, width, _ = annotated_image.shape

    # Loop through the detected hands to visualize.
    for _, hand_landmarks in enumerate(hand_landmarks_list):

        base = hand_landmarks[0]
        fingertips = [
            hand_landmarks[4],
            hand_landmarks[8],
            hand_landmarks[12],
            hand_landmarks[16],
            hand_landmarks[20],
        ]

        # Green color in BGR
        color = (255, 255, 255)

        # Line thickness of 9 px
        thickness = 4
        for finger in fingertips:
            p1 = (int(base.x * width), int(base.y * height))
            p2 = (int(finger.x * width), int(finger.y * height))
            cv2.line(annotated_image, p1, p2, color, thickness=thickness)

        # Get the top left corner of the detected hand's bounding box.
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        cv2.putText(
            annotated_image,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )
        # this is kind of black magic imo

        # Draw the hand landmarks.
    
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend(
            [
            
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in hand_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style(),
        )

    return annotated_image

# modified version of an answer from SO:
# https://stackoverflow.com/questions/57577445/list-available-cameras-opencv-python
def get_camera_status():
    """
    Test all camera ports and returns a tuple with the available ports
    and the ones that are working.
    """
    is_working = True
    dev_port = 0
    working_ports = []
    available_ports = []
    while is_working:
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            is_working = False
            print(f"Port {dev_port} is NOT WORKING.")
        else:
            # we don't care about the image, just if it's reading or not
            is_reading, _ = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print(f"Port {dev_port} is working and reads images ({h} x {w})")
                working_ports.append(dev_port)
            else:
                print(f"Port {dev_port} ({h} x {w} is present but DOES NOT READ)")
                available_ports.append(dev_port)
        dev_port += 1
    return available_ports, working_ports
