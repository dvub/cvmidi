"""maybe I should figure out how to prevent pylint from yelling at me for not having this"""

# pipenv run python main.py

# windows opencv debugging
# $env:OPENCV_LOG_LEVEL = "DEBUG"

# common imports
import time
import platform

from intro import intro
import util

# https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
import mediapipe as mp


# this is supposedly a fix for long VideoCapture open times on windows with Logi webcams.
# Didn't work for me though
# import os
# os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import mido


# I'm not sure why these were declared in google's code example
# but yeah
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


def main():
    midi_port, video_port = intro()

    # mediapipe configuration
    # Create a hand landmarker instance with the video mode:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=VisionRunningMode.VIDEO,
    )

    previous_value = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        # The landmarker is initialized. Use it here.
        # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap = cv2.VideoCapture(video_port)

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

            dist = util.calculate_distance(result)
            # translate the calculated distance to a midi CC value

            channel = 0  # MIDI channel (0-15)
            control_number = 7  # CC number (0-127)
            value = int(dist * 126)  # CC value (0-127)

            # TODO
            # discard minor +/-1 changes when the user is mostly not moving around
            if previous_value != value:
                print(f"{frame_timestamp}: NEW VALUE DETECTED.. sending MIDI message")
                cc_message = mido.Message(
                    "control_change",
                    channel=channel,
                    control=control_number,
                    value=value,
                )
                midi_port.send(cc_message)

            previous_value = value

            # draw all the fancy stuff on the image
            annotated_image = util.annotate_image(
                mp_image.numpy_view(), result, f"CC: {value}"
            )
            # show the annotated image
            cv2.imshow("Output", annotated_image)
            if cv2.waitKey(1) == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
