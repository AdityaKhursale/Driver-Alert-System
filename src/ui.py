import cv2

class UI:
    @staticmethod
    def resize(frame, dims):
        return cv2.resize(frame, dims)
    
    @staticmethod
    def draw_bbox(frame, topleft, bottomright, color, thickness=2):
        return cv2.rectangle(frame, topleft, bottomright, color, thickness=thickness)

    @staticmethod
    def draw_text(img, text, pos=(0, 0), font=cv2.FONT_HERSHEY_PLAIN,
        font_scale=3, text_color=(0, 255, 0), font_thickness=2,
        text_color_bg=(0, 0, 0)):
        x, y = pos
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        cv2.rectangle(img, pos, (x + text_w + 2, y + text_h + 7), text_color_bg, -1)
        cv2.putText(img, text, (x, y + text_h + int(font_scale) - 1), font, font_scale, text_color, font_thickness)

        return text_size
