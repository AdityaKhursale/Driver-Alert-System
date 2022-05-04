import cv2
import dlib
import os

from utils.decorators import ClassPropertyType, classproperty
from shared_data import MODELS_DIR
from imutils import face_utils


class FaceDescriptor:
    def __init__(self):
        self.face = None
        self.shapes = None
        self.shapes_orig = None


class FaceHelper(metaclass=ClassPropertyType):

    __face_detector = dlib.get_frontal_face_detector()
    __predictor_model_name = os.path.join(MODELS_DIR, "shape_predictor.dat")
    __predictor = dlib.shape_predictor(__predictor_model_name)

    @classmethod
    def get_faces(cls, img):
        faces = []
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fboxes = cls.face_detector(img, 0)
        for fbox in fboxes:
            new_face = FaceDescriptor()
            new_face.face = fbox
            new_face.shapes_orig, new_face.shapes = cls._get_predicted_shapes(img, fbox)
            faces.append(new_face)
        return faces

    @classproperty
    def face_detector(cls):
        return cls.__face_detector

    @classproperty
    def predictor_model_name(cls):
        return cls.__predictor_model_name

    @classproperty
    def predictor(cls):
        return cls.__predictor

    @staticmethod
    def get_bbox(face):
        face_tl = tuple((face.tl_corner().x, face.tl_corner().y))
        face_br = tuple((face.br_corner().x, face.br_corner().y))
        return face_tl, face_br

    @classmethod
    def _get_predicted_shapes(cls, img, face):
        prediction_orig = cls.predictor(img, face)
        prediction = face_utils.shape_to_np(prediction_orig)
        return prediction_orig, prediction
