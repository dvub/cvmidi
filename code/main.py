# pipenv run python main.py

# windows opencv debugging
# $env:OPENCV_LOG_LEVEL = "DEBUG"

import util
# common imports
import time
import math
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
from cv2 import cv2
import mido

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
    """
    hi
    """
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

def main():
    # SETUP

    print("Wecome to CV-MIDI.")
    print()
    print("MIDI PORT SETUP")
    print()
    # platform detection
    print("Detecting OS/platform...")
    system = platform.system()
    # system = "Windows"
    print(f"Your current platform seems to be: {system}")

    midi_port = 0

    if system == "Windows":
        print(
            "WARNING: Virtual MIDI ports are not supported by Windows natively. \n Please install a tool such as loopMIDI (https://www.tobias-erichsen.de/software/loopmidi.html) first."
        )

        print()

        print("Detecting current ports...")
        current_midi_inputs = mido.get_input_names()
        for index, value in enumerate(current_midi_inputs, start=1):
            print(f"{index}. {value}")

        # TODO
        # some validation is needed here
        num = int(input("Enter the port number you'd like to use:\n"))
        target_index = num - 1

        midi_port = mido.open_output(current_midi_inputs[target_index])

    else:
        print()
        # TODO
        # allow override here
        print("Creating a new Virtual MIDI Port. Is that OKAY?")

        midi_port = mido.open_output("CV-MIDI (Virtual)", virtual=True)

    print("VIDEO CAPTURE SETUP")
    print()
    list_ports()

    # mediapipe configuration
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a hand landmarker instance with the video mode:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=VisionRunningMode.VIDEO,
    )

    previous_value = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        # The landmarker is initialized. Use it here.
        # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap = cv2.VideoCapture(32)

        while True:
            # read the current frame
            _, frame = cap.read()
            frame = cv2.flip(frame, 1)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            # TODO:
            # I am almost certain this isn't the best way to get the current timestamp
            # fps = cap.get(cv2.CAP_PROP_FPS)
            frame_timestamp = round(time.time() * 1000)

            # actually do the processing with mediapipe!
            result = landmarker.detect_for_video(mp_image, frame_timestamp)

            dist = calculate_distance(result)
            # translate the calculated distance to a midi CC value

            channel = 0  # MIDI channel (0-15)
            control_number = 7  # CC number (0-127)
            value = int(dist * 126)  # CC value (0-127)

            # TODO
            # discard minor +/-1 changes when the user is mostly not moving around
            if previous_value != value:
                print(f"{frame_timestamp}: NEW VALUE DETECTED.. sending MIDI message")
                cc_message = mido.Message(
                    "control_change", channel=channel, control=control_number, value=value
                )
                midi_port.send(cc_message)

            previous_value = value

            # draw all the fancy stuff on the image
            annotated_image = annotate_image(mp_image.numpy_view(), result, f"CC: {value}")
            # show the annotated image
            cv2.imshow("Output", annotated_image)
            if cv2.waitKey(1) == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__": 
    main()
    
