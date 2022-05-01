import cv2

class UI:
    @staticmethod
    def resize(frame, dims):
        return cv2.resize(frame, dims)
    
    @staticmethod
    def draw_bbox(frame, topleft, bottomright, color):
        return cv2.rectangle(frame, topleft, bottomright, color)