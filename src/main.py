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

    def run(self, webcam=0):
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
            # TODO: Change size dynamically
            self.ui.resize(frame, (1300, 800))
            faces = self.face_helper.get_faces(gray)

            cv2.putText(frame, "Eyes:", (0, 700), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        ColorPalette.whiteColor.value, 2)
            for face in faces:
                self.ui.draw_bbox(frame, *self.face_helper.get_bbox(face), ColorPalette.whiteColor.value)
                l_eye, r_eye = self.eye_helper.get_eyes(gray, face)
                self.ui.draw_bbox(frame, *self.eye_helper.get_bbox(l_eye), ColorPalette.greenColor.value)
                self.ui.draw_bbox(frame, *self.eye_helper.get_bbox(r_eye), ColorPalette.greenColor.value)

                # TODO: Modularize more, move this part to drowsiness
                # Once other features are up of drowsiness from Anurag
                if (self.eye_helper.is_eye_closed(l_eye)
                    and self.eye_helper.is_eye_closed(r_eye)):
                    cv2.putText(frame, "Closed", (85, 702),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, ColorPalette.redColor.value, 2)
                    counter += 1
                    if counter >= self.eye_helper.SLEEP_CONSEC_FR_THRESH:
                        counter = 0
                        t = Thread(target=SoundAlert.play_drowsiness_alert)
                        t.daemon = True
                        t.start()
                        drowsinessAlertSet = 20
                else:
                    counter = 0
                    cv2.putText(frame, "Open", (85, 702),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, ColorPalette.greenColor.value, 2)
                if drowsinessAlertSet > 0:
                    cv2.putText(frame, "Drowsiness Alert", (0, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 3, ColorPalette.redColor.value, 3)
                    drowsinessAlertSet -= 1
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
