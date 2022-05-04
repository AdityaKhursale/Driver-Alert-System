from numpy import eye
import cv2
import dlib
import logging
import os

from collections import namedtuple
from imutils import face_utils
from scipy.spatial import distance as dist

from utils.decorators import Cached, ClassPropertyType, classproperty
from vision.face_helper import FaceHelper


logger = logging.getLogger("driver_alert")


class EyeHelper(metaclass=ClassPropertyType):
    
    # Open EYE Aspect Ratio Threshold
    # If ratio < thresh --> close, open otherwise
    EYE_AR_THRESH = 0.18
    SLEEP_CONSEC_FR_THRESH = 20

    EyePosition = namedtuple("EyePosition", "start end")
    
    @staticmethod
    def get_aspect_ratio(eye_marks):
        if len(eye_marks) < 6:
            logger.info(0.0)
            return 0.0
        v_distance_1 = dist.euclidean(eye_marks[1], eye_marks[5])
        v_distance_2 = dist.euclidean(eye_marks[2], eye_marks[4])

        h_distance = dist.euclidean(eye_marks[0], eye_marks[3])

        logger.info((v_distance_1 + v_distance_2) / (2.0 * h_distance))
        return (v_distance_1 + v_distance_2) / (2.0 * h_distance)

    @classmethod
    def is_eye_closed(cls, eye_marks):
        return cls.get_aspect_ratio(eye_marks) < cls.EYE_AR_THRESH
    
    @classproperty
    def left_eye_pos(cls):
        return cls.EyePosition(face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][0],
                               face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][1])
    
    @classproperty
    def right_eye_pos(cls):
        return cls.EyePosition(face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][0],
                              face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][1])

    @classmethod
    def get_eyes(cls, img, face):
         # TODO: Make it accept non grayed image
        # without need to gray for faces and eyes both
        pred = FaceHelper._get_predicted_shapes(img, face)
        left_eye = pred[cls.left_eye_pos.start:cls.left_eye_pos.end]
        right_eye = pred[cls.right_eye_pos.start:cls.right_eye_pos.end]
        return cv2.convexHull(left_eye), cv2.convexHull(right_eye)
    
    @classmethod
    def get_left_eye(cls, img, face):
         # TODO: Make it accept non grayed image
        # without need to gray for faces and eyes both
        pred = FaceHelper._get_predicted_shapes(img, face)
        left_eye = pred[cls.left_eye_pos.start:cls.left_eye_pos.end]
        return cv2.convexHull(left_eye)
    
    @classmethod
    def get_right_eye(cls, img, face):
         # TODO: Make it accept non grayed image
        # without need to gray for faces and eyes both
        pred = FaceHelper._get_predicted_shapes(img, face)
        right_eye = pred[cls.right_eye_pos.start:cls.right_eye_pos.end]
        return cv2.convexHull(right_eye)

    @classmethod
    def get_bbox(cls, eye):
        bbox = cv2.boundingRect(eye)
        tl = tuple((bbox[0], bbox[1]))
        br = tuple ((bbox[0]+bbox[2], bbox[1]+bbox[3]))
        return tl, br
    