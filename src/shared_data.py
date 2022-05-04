import os

from collections import namedtuple
from enum import Enum


ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
CONFIG_DIR = os.path.join(ROOT_DIR, "configs")
RESOURCES_DIR = os.path.join(ROOT_DIR, "resources")
MODELS_DIR = os.path.join(ROOT_DIR, "models")


Color = namedtuple("Color", "B G R")
FacialmarkPos = namedtuple("Position", "start end")

class ColorPalette(Enum):
    grayColor = Color(100, 100, 100)
    darkgrayColor = Color(169, 169, 169)
    greenColor = Color(0, 255, 0)
    redColor = Color(0, 0, 255)
    whiteColor = Color(255, 255, 255)
    orangeColor = Color(0, 165, 255)
    yellowColor = Color(0, 255, 255)
    blueColor = Color(255, 0, 0)
    blackColor = Color(0, 0, 0)