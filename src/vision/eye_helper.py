import cv2
import dlib
import os

from collections import namedtuple
from imutils import face_utils
from scipy.spatial import distance as dist
from shared_data import MODELS_DIR
from utils.decorators import Cached, ClassPropertyType, classproperty


class EyeHelper(metaclass=ClassPropertyType):
    
    # Open EYE Aspect Ratio Threshold
    # If ratio < thresh --> close, open otherwise
    EYE_AR_THRESH = 0.25
    SLEEP_CONSEC_FR_THRESH = 60
    
    __predictor_model_name = os.path.join(MODELS_DIR, "shape_predictor.dat")

    EyePosition = namedtuple("EyePosition", "start end")
    
    @staticmethod
    def get_aspect_ratio(eye_marks):
        if len(eye_marks) == 6:
            v_distance_1 = dist.euclidean(eye_marks[1], eye_marks[5])
            v_distance_2 = dist.euclidean(eye_marks[2], eye_marks[4])

            h_distance = dist.euclidean(eye_marks[0], eye_marks[3])

            return (v_distance_1 + v_distance_2) / (2.0 * h_distance)
        else:
            v_distance_1 = dist.euclidean(eye_marks[2], eye_marks[4])
            h_distance = dist.euclidean(eye_marks[0], eye_marks[3])
            return v_distance_1 / h_distance
    
    @classmethod
    def is_eye_closed(cls, eye_marks):
        return cls.get_aspect_ratio(eye_marks) < cls.EYE_AR_THRESH
    
    @classproperty
    def predictor_model_name(cls):
        return cls.__predictor_model_name

    @classproperty
    def predictor(cls):
        return dlib.shape_predictor(cls.predictor_model_name)

    @classproperty
    def left_eye_pos(cls):
        return cls.EyePosition(face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][0],
                               face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][1])
    
    @classproperty
    def right_eye_pos(cls):
        return cls.EyePosition(face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][0],
                              face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][1])


    # TODO: Add @Cached once caching based on parameters is done
    # TODO: Update get eye based on that to avoid code duplication
    @classmethod
    def _get_predicted_eyes(cls, frame, face):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prediction = cls.predictor(gray, face)
        prediction = face_utils.shape_to_np(prediction)
        return prediction

    @classmethod
    def get_eyes(cls, frame, face):
        pred = cls._get_predicted_eyes(frame, face)
        left_eye = pred[cls.left_eye_pos.start:cls.left_eye_pos.end]
        right_eye = pred[cls.right_eye_pos.start:cls.right_eye_pos.end]
        return cv2.convexHull(left_eye), cv2.convexHull(right_eye)
    
    @classmethod
    def get_left_eye(cls, frame, face):
        pred = cls._get_predicted_eyes(frame, face)
        left_eye = pred[cls.left_eye_pos.start:cls.left_eye_pos.end]
        return cv2.convexHull(left_eye)
    
    @classmethod
    def get_right_eye(cls, frame, face):
        pred = cls._get_predicted_eyes(frame, face)
        right_eye = pred[cls.right_eye_pos.start:cls.right_eye_pos.end]
        return cv2.convexHull(right_eye)

    @classmethod
    def get_bbox(cls, eye):
        bbox = cv2.boundingRect(eye)
        tl = tuple((bbox[0], bbox[1]))
        br = tuple ((bbox[0]+bbox[2], bbox[1]+bbox[3]))
        return tl, br
    