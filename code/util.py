# common imports
import platform
import numpy as np


# https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# this is supposedly a fix for long VideoCapture open times on windows with Logi webcams.
# Didn't work for me though
# import os
# os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


# TODO
# this should potentially return an array
# if the array is empty, no hands are present, etc.
def calculate_distance(detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness

    dist_sum = 0
    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]

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
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)
    height, width, _ = annotated_image.shape

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]

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


# https://stackoverflow.com/questions/57577445/list-available-cameras-opencv-python
def list_ports():
    """
    Test the ports and returns a tuple with the available ports
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
            print("Port %s is not working." % dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print(
                    "Port %s is working and reads images (%s x %s)" % (dev_port, h, w)
                )
                working_ports.append(dev_port)
            else:
                print(
                    "Port %s for camera ( %s x %s) is present but does not reads."
                    % (dev_port, h, w)
                )
                available_ports.append(dev_port)
        dev_port += 1
    return available_ports, working_ports
