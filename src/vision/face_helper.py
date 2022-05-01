import cv2
import dlib # TODO: Update requirements


from utils.decorators import Cached, ClassPropertyType, classproperty


class FaceHelper(metaclass=ClassPropertyType):

    @classmethod
    def get_faces(cls, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fboxes = cls.face_detector(gray, 0)
        return fboxes

    @classproperty
    def face_detector(self):
        return dlib.get_frontal_face_detector()
    
    @staticmethod
    def get_bbox(face):
        face_tl = tuple((face.tl_corner().x, face.tl_corner().y))
        face_br = tuple((face.br_corner().x, face.br_corner().y))
        return face_tl, face_br