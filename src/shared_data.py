import os

from collections import namedtuple

ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
CONFIG_DIR = os.path.join(ROOT_DIR, "configs")
RESOURCES_DIR = os.path.join(ROOT_DIR, "resources")
MODELS_DIR = os.path.join(ROOT_DIR, "models")


Color = namedtuple("Color", "B G R")
grayColor = Color(100, 100, 100)
greenColor = Color(0, 255, 0)
redColor = Color(0, 0, 255)


FACIAL_LANDMARK_IDENTIFIER = os.path.join(MODELS_DIR,)