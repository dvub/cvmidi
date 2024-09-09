# $env:OPENCV_LOG_LEVEL = "DEBUG"
# pipenv run python main.py

# https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
# https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# import os
# os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import time
import math
import numpy as np

import mido

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

# this very nice function came from here
# https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#scrollTo=_JVO3rvPD4RN&uniqifier=1


def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)
    height, width, _ = annotated_image.shape

    dist = 0

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]

        thumb = hand_landmarks[4]
        index_finger = hand_landmarks[8]

        dist = math.sqrt(
            pow(index_finger.x - thumb.x, 2)
            + pow(index_finger.y - thumb.y, 2)
            + pow(index_finger.z - thumb.z, 2)
        )

        # Green color in BGR
        color = (0, 255, 0)

        # Line thickness of 9 px
        thickness = 9

        p1 = (int(thumb.x * width), int(thumb.y * height))
        p2 = (int(index_finger.x * width), int(index_finger.y * width))
        cv2.line(annotated_image, p1, p2, color, thickness=thickness)

        # Get the top left corner of the detected hand's bounding box.
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(
            annotated_image,
            f"{dist}",
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

    return annotated_image, dist


# MIDI
print(mido.get_output_names())
outport = mido.open_output("loopMIDI Port 1")


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the video mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
)


with HandLandmarker.create_from_options(options) as landmarker:
    # The landmarker is initialized. Use it here.
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # cap = cv2.VideoCapture(0)

    while True:
        # read the current frame
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # TODO:
        # I am almost certain this isn't the best way to get the current timestamp
        # fps = cap.get(cv2.CAP_PROP_FPS)
        frame_timestamp = round(time.time() * 1000)

        result = landmarker.detect_for_video(mp_image, frame_timestamp)

        annotated_image, dist = draw_landmarks_on_image(mp_image.numpy_view(), result)

        channel = 0  # MIDI channel (0-15)
        control_number = 7  # CC number (0-127)
        value = min(126, max(int(dist * 127), 0))  # CC value (0-127)
        print(value)
        cc_message = mido.Message(
            "control_change", channel=channel, control=control_number, value=value
        )
        outport.send(cc_message)

        cv2.imshow("Output", annotated_image)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
