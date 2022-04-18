import os
import logging

from playsound import playsound
from shared_data import RESOURCES_DIR

logger = logging.getLogger("driver_alert")

class Alert:
    @staticmethod
    def alert_drowsiness():
        playsound(os.path.join(RESOURCES_DIR, "sound.wav"))
    
