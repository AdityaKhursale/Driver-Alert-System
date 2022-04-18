import argparse
import cv2
import dlib
import glob
import logging
import logging.config
import os
import sys
import time
import traceback
import yaml

from alert import Alert
from eye_helper import Eye
from imutils import face_utils
from imutils import resize
from imutils.video import VideoStream
from shared_data import CONFIG_DIR, grayColor, greenColor, redColor
from threading import Thread
from utils import remove_files

args = None
global logger


def run():
    counter = 0
    drowsy_alarm_on = False

    # TODO: Modularize below code

    logger.debug("loading face detector")
    detector = dlib.get_frontal_face_detector()
    logger.debug("loading shape predictor")
    predictor = dlib.shape_predictor(Eye.predictor_model)

    l_e_start, l_e_end = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    r_e_start, r_e_end = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    logger.info("starting camera capture")
    stream = VideoStream(args.webcam).start()
    time.sleep(1.0)

    frame_no = 0
    while True:
        frame_no += 1
        # TODO: Revisit and convert this to full window
        logger.debug("frame number: {}".format(frame_no))
        frame = stream.read()
        frame = resize(frame, width=450) 

        logger.debug("converting to grayscale")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        logger.debug("detecting faces")
        fboxes = detector(gray, 0)

        for fbox in fboxes:
            # TODO: Check if these need to be changed to namedtuple
            face_tl = tuple((fbox.tl_corner().x, fbox.tl_corner().y))
            face_br = tuple((fbox.br_corner().x, fbox.br_corner().y))
            logger.debug("face detected, top_left={}, bottom_right={}".format(
                face_tl, face_br))

            logger.debug("marking face over frame")
            cv2.rectangle(frame, face_tl, face_br, grayColor)

            logger.debug("detecting eyes")
            shape = predictor(gray, fbox)
            shape = face_utils.shape_to_np(shape)

            l_eye = shape[l_e_start:l_e_end]
            r_eye = shape[r_e_start:r_e_end]
            
            l_e_hull = cv2.convexHull(l_eye)
            r_e_hull = cv2.convexHull(r_eye)
            
            l_e_box = cv2.boundingRect(l_e_hull)
            r_e_box = cv2.boundingRect(r_e_hull)
            
            l_e_tl = tuple((l_e_box[0], l_e_box[1]))
            l_e_br = tuple((l_e_box[0]+l_e_box[2], l_e_box[1]+l_e_box[3]))

            r_e_tl = tuple((r_e_box[0], r_e_box[1]))
            r_e_br = tuple((r_e_box[0]+r_e_box[2], r_e_box[1]+r_e_box[3]))

            logger.debug("eyes detected, "
                          "left_eye_top_left={}, left_eye_bottom_right={}, "
                          "right_eye_top_left={}, right_eye_bottom_right={}".format(
                              l_e_tl, l_e_br, r_e_tl, r_e_br))

            logger.debug("marking eyes")
            cv2.rectangle(frame, l_e_tl, l_e_br, greenColor)
            cv2.rectangle(frame, r_e_tl, r_e_br, greenColor)

            logger.debug("Checking if driver is sleepy")
            if Eye.is_eye_open(l_eye) and Eye.is_eye_open(r_eye):
                counter += 1
                logger.debug("Eyes closed for consecutive {} frames".format(counter))

                if counter >= Eye.SLEEP_CONSEC_FR_THRESH:
                    logger.debug("driver is sleepy")
                    logger.debug("playing drowsy alarm")
                    if not drowsy_alarm_on:
                        drowsy_alarm_on = True
                    
                    t = Thread(target=Alert.alert_drowsiness)
                    t.daemon = True
                    t.start()
                
                    # TODO: Beautify and align on the UI
                    logger.debug("Setting sleepy text as Yes")
                    cv2.putText(frame, "Sleepy: Yes", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, redColor, 2)
            else:
                counter = 0
                drowsy_alarm_on = False
    
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    stream.stop()
    
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
    run()
