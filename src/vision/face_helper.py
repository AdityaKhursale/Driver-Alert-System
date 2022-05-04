import os
import dlib  # TODO: Update requirements


from utils.decorators import Cached, ClassPropertyType, classproperty
from shared_data import MODELS_DIR
from imutils import face_utils


class FaceHelper(metaclass=ClassPropertyType):

    __face_detector = dlib.get_frontal_face_detector()
    __predictor_model_name = os.path.join(MODELS_DIR, "shape_predictor.dat")
    __predictor = dlib.shape_predictor(__predictor_model_name)

    @classmethod
    def get_faces(cls, img):
        # TODO: Make it accept non grayed image
        # without need to gray for faces and eyes both
        fboxes = cls.face_detector(img, 0)
        return fboxes

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

    # TODO: Add @Cached once caching based on parameters is done
    # TODO: Update get eye based on that to avoid code duplication
    @classmethod
    def _get_predicted_shapes(cls, img, face):
         # TODO: Make it accept non grayed image
        # without need to gray for faces and eyes both
        prediction = cls.predictor(img, face)
        prediction = face_utils.shape_to_np(prediction)
        return prediction
