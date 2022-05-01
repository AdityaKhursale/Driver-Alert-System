import os
import logging

from playsound import playsound
from shared_data import RESOURCES_DIR


logger = logging.getLogger("driver_alert")


class SoundAlert:
    @staticmethod
    def play_drowsiness_alert():
        playsound(os.path.join(RESOURCES_DIR, "you_suffer.mp3"))
