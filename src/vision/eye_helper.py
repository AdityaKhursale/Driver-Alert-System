import cv2
import logging

from collections import namedtuple
from imutils import face_utils
from scipy.spatial import distance as dist
from shared_data import FacialmarkPos
from utils.decorators import ClassPropertyType, classproperty


logger = logging.getLogger("driver_alert")


class EyeHelper(metaclass=ClassPropertyType):
    
    EYE_AR_THRESH = 0.18  # Open eye threshold
    SLEEP_CONSEC_FR_THRESH = 8

    @staticmethod
    def get_aspect_ratio(eye_marks):
        if len(eye_marks) < 6:
            # logger.info(0.0)
            return 0.0
        v_distance_1 = dist.euclidean(eye_marks[1], eye_marks[5])
        v_distance_2 = dist.euclidean(eye_marks[2], eye_marks[4])

        h_distance = dist.euclidean(eye_marks[0], eye_marks[3])

        # logger.info((v_distance_1 + v_distance_2) / (2.0 * h_distance))
        return (v_distance_1 + v_distance_2) / (2.0 * h_distance)

    @classmethod
    def is_eye_closed(cls, eye_marks):
        ratio = cls.get_aspect_ratio(eye_marks)
        return ratio < cls.EYE_AR_THRESH, ratio
    
    @classproperty
    def left_eye_pos(cls):
        return FacialmarkPos(face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][0],
                             face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][1])

    @classproperty
    def right_eye_pos(cls):
        return FacialmarkPos(face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][0],
                             face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][1])

    @classmethod
    def get_eyes(cls, shapes):
        return cls.get_left_eye(shapes), cls.get_right_eye(shapes)
    
    @classmethod
    def get_left_eye(cls, shapes):
        left_eye = shapes[cls.left_eye_pos.start:cls.left_eye_pos.end]
        return cv2.convexHull(left_eye)
    
    @classmethod
    def get_right_eye(cls, shapes):
        right_eye = shapes[cls.right_eye_pos.start:cls.right_eye_pos.end]
        return cv2.convexHull(right_eye)

    @classmethod
    def get_bbox(cls, eye):
        bbox = cv2.boundingRect(eye)
        tl = tuple((bbox[0], bbox[1]))
        br = tuple ((bbox[0]+bbox[2], bbox[1]+bbox[3]))
        return tl, br
    