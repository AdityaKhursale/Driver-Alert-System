import os

from scipy.spatial import distance as dist
from shared_data import MODELS_DIR
from utils import ClassPropertyType, classproperty


class Eye(metaclass= ClassPropertyType):
    # Open EYE Aspect Ratio Threshold
    # If ratio < thresh --> close, open otherwise
    EYE_AR_THRESH = 0.3
    SLEEP_CONSEC_FR_THRESH = 100
    
    __predictor_model = os.path.join(MODELS_DIR, "shape_predictor.dat")

    @staticmethod
    def get_aspect_ratio(eye_marks):
        v_distance_1 = dist.euclidean(eye_marks[1], eye_marks[5])
        v_distance_2 = dist.euclidean(eye_marks[2], eye_marks[4])

        h_distance = dist.euclidean(eye_marks[0], eye_marks[3])

        return (v_distance_1 + v_distance_2) / (2.0 * h_distance)
    
    @classmethod
    def is_eye_open(cls, eye_marks):
        return cls.get_aspect_ratio(eye_marks) < cls.EYE_AR_THRESH
    
    @classproperty
    def predictor_model(cls):
        return cls.__predictor_model
