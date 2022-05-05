import logging
import numpy as np

from utils.decorators import ClassPropertyType, classproperty
from scipy.spatial import distance as dist
from shared_data import FacialmarkPos


logger = logging.getLogger("driver_alert")


class MouthHelper(metaclass=ClassPropertyType):
    
    MOUTH_AR_THRESH = 60

    @classproperty
    def lip_boundary_pos(cls):
        return FacialmarkPos(48, 60)
    
    @classproperty
    def upper_lip_pos(cls):
        return FacialmarkPos(50, 53)
    
    @classproperty
    def upper_inner_mouth(cls):
        return FacialmarkPos(61, 64)
    
    @classproperty
    def lower_lip_pos(cls):
        return FacialmarkPos(56, 59)
    
    @classproperty
    def lower_inner_mouth(cls):
        return FacialmarkPos(65, 68)

    @classmethod
    def get_lip_boundary(cls, shapes):
        return shapes[cls.lip_boundary_pos.start:cls.lip_boundary_pos.end]
    
    @classmethod
    def get_upper_lip(cls, shapes):
        return shapes[cls.upper_lip_pos.start:cls.upper_lip_pos.end]
    
    @classmethod
    def get_lower_lip(cls, shapes):
        return shapes[cls.lower_lip_pos.start:cls.lower_lip_pos.end]
    
    @classmethod
    def get_upper_inner_mouth(cls, shapes):
        return shapes[cls.upper_inner_mouth.start:cls.upper_inner_mouth.end]
    
    @classmethod
    def get_lower_inner_mouth(cls, shapes):
        return shapes[cls.lower_inner_mouth.start:cls.lower_inner_mouth.end]
    
    @classmethod
    def get_aspect_ratio(cls, shapes):
        upper_mouth = np.concatenate((cls.get_upper_lip(shapes),
                                      cls.get_upper_inner_mouth(shapes)))
        lower_mouth = np.concatenate((cls.get_lower_lip(shapes),
                                      cls.get_lower_inner_mouth(shapes)))
        return dist.euclidean(np.mean(upper_mouth, axis=0),
                              np.mean(lower_mouth, axis=0))

    @classmethod
    def is_mouth_open(cls, shapes):
        ratio = cls.get_aspect_ratio(shapes)
        return ratio > cls.MOUTH_AR_THRESH, ratio
    