import argparse
import cv2

import logging
import logging.config
import os
import sys
import time
import traceback
import yaml

from alert.sound_alert import SoundAlert
from ui import UI
from imutils.video import VideoStream
from shared_data import CONFIG_DIR, Color, ColorPalette
from threading import Thread
from utils.misc import remove_files
from vision.eye_helper import EyeHelper
from vision.face_helper import FaceHelper
from vision.mouth_helper import MouthHelper
from vision.eye import Eye
from vision.calibration import Calibration

# Emotion Imports
import numpy as np
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

args = None
global logger


class DriverAlertSystem:
    def __init__(self):
        self.initialize_modules()
        self.stream = None
    
    def initialize_modules(self):
        self.ui = UI()
        self.eye_helper = EyeHelper()
        self.face_helper = FaceHelper()
        self.mouth_helper = MouthHelper()

    def run(self, webcam=0):
        """
        Emotion Recognition
        """
        emotion_model_path = './models/emotion_model.hdf5'
        emotion_labels = get_labels('fer2013')

        frame_window = 10
        emotion_offsets = (20, 40)

        face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
        emotion_classifier = load_model(emotion_model_path)

        emotion_target_size = emotion_classifier.input_shape[1:3]
        emotion_window = []

        logger.info("starting camera capture")

        self.stream = VideoStream(webcam).start()
        time.sleep(1.0) # Wait for stream to be started

        frame_no = 0
        counter = 0
        drowsinessAlertSet = 0
        while True:
            frame_no += 1
            logger.debug("Frame number: {}".format(frame_no))
            frame = self.stream.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_helper.get_faces(gray)

            efaces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
			            minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

            self.ui.draw_text(frame, "Distracted Detection", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, ColorPalette.whiteColor.value, 1,
                        ColorPalette.blackColor.value)
            cv2.putText(frame, "Eyes:", (0, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        ColorPalette.whiteColor.value, 1)
            cv2.putText(frame, "Mouth:", (200, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        ColorPalette.whiteColor.value, 1)
            cv2.putText(frame, "Head Tilt:", (380, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        ColorPalette.whiteColor.value, 1)
            cv2.putText(frame, "Left pupil:  ",
                        (90, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        ColorPalette.whiteColor.value, 1)
            cv2.putText(frame, "Right pupil: ",
                        (90, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        ColorPalette.whiteColor.value, 1)
            self.ui.draw_text(frame, "Drowsiness Detection", (0, 650), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, ColorPalette.whiteColor.value, 1,
                        ColorPalette.blackColor.value)
            self.ui.draw_text(frame, "Emotion Recognition", (850, 650), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, ColorPalette.whiteColor.value, 1,
                        ColorPalette.blackColor.value)
            cv2.putText(frame, "Driver Mood:", (900, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        ColorPalette.whiteColor.value, 1)
            for face in faces:
                self.ui.draw_bbox(frame, *self.face_helper.get_bbox(face.face), ColorPalette.whiteColor.value)
                l_eye, r_eye = self.eye_helper.get_eyes(face.shapes)
                self.ui.draw_bbox(frame, *self.eye_helper.get_bbox(l_eye), ColorPalette.greenColor.value)
                self.ui.draw_bbox(frame, *self.eye_helper.get_bbox(r_eye), ColorPalette.greenColor.value)

                # TODO: Modularize more, move this part to drowsiness
                # Once other features are up of drowsiness from Anurag
                """
                Sleepiness check
                """

                if (self.eye_helper.is_eye_closed(l_eye)
                    and self.eye_helper.is_eye_closed(r_eye)):
                    cv2.putText(frame, "Closed", (61, 702),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, ColorPalette.redColor.value, 2)
                    counter += 1
                    if counter >= self.eye_helper.SLEEP_CONSEC_FR_THRESH:
                        counter = 0
                        t = Thread(target=SoundAlert.play_drowsiness_alert)
                        t.daemon = True
                        t.start()
                        drowsinessAlertSet = 20
                else:
                    counter = 0
                    cv2.putText(frame, "Open", (61, 702),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, ColorPalette.greenColor.value, 2)
                if drowsinessAlertSet > 0:
                    cv2.putText(frame, "Driver is Sleeping",
                                (frame.shape[1] // 2 - 210, frame.shape[0] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, ColorPalette.redColor.value, 4)
                    drowsinessAlertSet -= 1

                """
                Yawning check
                """
                cv2.drawContours(frame,
                                 [self.mouth_helper.get_lip_boundary(face.shapes)],
                                 -1, ColorPalette.orangeColor.value, thickness=2)
                if self.mouth_helper.is_mouth_open(face.shapes):
                    cv2.putText(frame, "Open", (275, 702),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, ColorPalette.redColor.value, 2)
                    cv2.putText(frame, "Driver is Yawning",
                                (frame.shape[1] // 2 - 210, frame.shape[0] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, ColorPalette.redColor.value, 4)
                else:
                    cv2.putText(frame, "Closed", (275, 702),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, ColorPalette.greenColor.value, 2)

                """
                Gaze Tracking
                """
                eye_left = Eye(gray, face.shapes_orig, 0, Calibration())
                eye_right = Eye(gray, face.shapes_orig, 1, Calibration())
                x_left = eye_left.origin[0] + eye_left.pupil.x
                y_left = eye_left.origin[1] + eye_left.pupil.y
                x_right = eye_right.origin[0] + eye_right.pupil.x
                y_right = eye_right.origin[1] + eye_right.pupil.y
                cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), ColorPalette.blueColor.value)
                cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), ColorPalette.blueColor.value)
                cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), ColorPalette.blueColor.value)
                cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), ColorPalette.blueColor.value)

                pupil_left = eye_left.pupil.x / (eye_left.center[0] * 2 - 10)
                pupil_right = eye_right.pupil.x / (eye_right.center[0] * 2 - 10)
                h_ratio = (pupil_left + pupil_right) / 2

                pupil_left = eye_left.pupil.y / (eye_left.center[1] * 2 - 10)
                pupil_right = eye_right.pupil.y / (eye_right.center[1] * 2 - 10)
                v_ratio = (pupil_left + pupil_right) / 2

                text = ""
                if h_ratio <= 0.45:
                    text = "Looking right"
                    cv2.putText(frame, text, (1000, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                ColorPalette.redColor.value, 2)
                elif h_ratio >= 0.75:
                    text = "Looking left"
                    cv2.putText(frame, text, (90, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                ColorPalette.redColor.value, 2)
                elif h_ratio > 0.45 and h_ratio < 0.75:
                    text = "Looking center"
                    cv2.putText(frame, text, (500, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                ColorPalette.greenColor.value, 2)
                left_pupil = x_left, y_left
                right_pupil = x_right, y_right
                cv2.putText(frame, str(left_pupil),
                            (183, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            ColorPalette.blueColor.value, 1)
                cv2.putText(frame, str(right_pupil),
                            (183, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            ColorPalette.blueColor.value, 1)

                """
                Emotion Recognition
                """
                for face_coordinates in efaces:

                    x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
                    gray_face = gray[y1:y2, x1:x2]
                    try:
                        gray_face = cv2.resize(gray_face, (emotion_target_size))
                    except:
                        continue

                    gray_face = preprocess_input(gray_face, True)
                    gray_face = np.expand_dims(gray_face, 0)
                    gray_face = np.expand_dims(gray_face, -1)
                    emotion_prediction = emotion_classifier.predict(gray_face)
                    emotion_probability = np.max(emotion_prediction)
                    emotion_label_arg = np.argmax(emotion_prediction)
                    emotion_text = emotion_labels[emotion_label_arg]
                    emotion_window.append(emotion_text)

                    if len(emotion_window) > frame_window:
                        emotion_window.pop(0)
                    try:
                        emotion_mode = mode(emotion_window)
                    except:
                        continue

                    if emotion_text == 'angry':
                        color = ColorPalette.redColor.value
                    elif emotion_text == 'sad':
                        color = ColorPalette.redColor.value
                    elif emotion_text == 'happy':
                        color = ColorPalette.greenColor.value
                    elif emotion_text == 'surprise':
                        color = ColorPalette.blueColor.value
                    else:
                        color = ColorPalette.yellowColor.value
                    cv2.putText(frame, emotion_mode.capitalize(), (1050, 702),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, color, 2)
            cv2.imshow("Driver Alert System", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

        cv2.destroyAllWindows()
        self.stream.stop()


def populate_args():
    global args
    parser = argparse.ArgumentParser(description="Driver Alert System")
    parser.add_argument("--logsdir", required=False, default=".",
                        help="directory to store logs")
    parser.add_argument("--webcam", required=False, default=0,
                        help="camera device id to use")
    args = parser.parse_args()


def setup_logging():
    # TODO: Update log handler to move logfile to logsdir
    try:
        if not os.path.exists(args.logsdir):
            os.makedirs(args.logsdir)
    except IOError:
        print("Failed to create directory for logs")
        print(str(traceback.format_exc()))
        sys.exit(1)

    remove_files(args.logsdir, "driver_alert_debug.log*")

    global logger
    config_file = os.path.join(CONFIG_DIR, "logging_config.yml")
    with open(config_file, "r") as f:
        logging_config = yaml.safe_load(f.read())

    logging.config.dictConfig(logging_config)
    logger = logging.getLogger("driver_alert")


def setup():
    populate_args()
    setup_logging()


if __name__ == "__main__":
    setup()
    dalertsystem = DriverAlertSystem()
    dalertsystem.run(args.webcam)
