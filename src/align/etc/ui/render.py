import cv2

from enum import Enum

import numpy as np

class FACESTATE(Enum):
    DETECTED       = 1
    NO_MATCH_FOUND = 2
    MATCH_FOUND    = 3

class Renderer:

    font = cv2.FONT_HERSHEY_DUPLEX
    current_frame = None

    blue_normal = (0, 113, 197)
    blue_dark = (0, 60, 113)
    blue_sky = (0, 174, 239)
    white = (255, 255, 255)
    yellow = (243, 213, 78)
    green = (196, 214, 0)

    def __init__(self, camera_id=0, window_name="AIOffload"):
        self.window_name = window_name
        self.detected_img = cv2.imread('etc/ui/pixil-frame-tred.png', -1)
        self.recog_no_match_img = cv2.imread('etc/ui/pixil-frame-tgreen.png', -1)
        self.recog_match_img = cv2.imread('etc/ui/pixil-frame-tnblue.png', -1)
        self.text_background = cv2.imread('etc/ui/pixil-frame-text2.png', -1)

    def draw_face_bounding_box(self, x, y, w, h, curr_state):
        if curr_state == FACESTATE.DETECTED:
            s_img = cv2.resize(self.detected_img, (w, h))
        elif curr_state == FACESTATE.NO_MATCH_FOUND:
            s_img = cv2.resize(self.recog_no_match_img, (w, h))
        else:
            s_img = cv2.resize(self.recog_match_img, (w, h))

        self.draw_image_overlay_alpha(x, y, s_img)

    def draw_text_labels(self, x, y, w, h, tuple_first_match, tuple_second_match, tuple_third_match):
        s_img = cv2.resize(self.text_background, (200, 120))
        y = y + h + 2

        if self.draw_image_overlay_alpha(x, y, s_img) is False:
            return

        cv2.putText(self.current_frame, tuple_first_match[0], (x+2, y + 20), self.font, 0.5, self.white)
        self.draw_image_overlay(x+165, y - 5, cv2.resize(tuple_first_match[1], (40, 40)))

        cv2.putText(self.current_frame, tuple_second_match[0], (x+2, y + 65), self.font, 0.5, self.white)
        self.draw_image_overlay(x+165, y + 45 - 5, cv2.resize(tuple_second_match[1], (40, 40)))

        cv2.putText(self.current_frame, tuple_third_match[0], (x+2, y + 105), self.font, 0.5, self.white)
        self.draw_image_overlay(x+165, y + 90 - 5, cv2.resize(tuple_third_match[1], (40, 40)))

    def draw_image_overlay_alpha(self, x, y, s_img):
        if y <= 0 or y + s_img.shape[0] >= self.current_frame.shape[0]:
            return False
        if x <= 0 or x + s_img.shape[1] >= self.current_frame.shape[1]:
            return False

        y1, y2 = y, y + s_img.shape[0]
        x1, x2 = x, x + s_img.shape[1]
        alpha_s = s_img[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            self.current_frame[y1:y2, x1:x2, c] = (
                        alpha_s * s_img[:, :, c] + alpha_l * self.current_frame[y1:y2, x1:x2, c])

    def draw_image_overlay(self, x_offset, y_offset, img):
        if y_offset + img.shape[0] >= self.current_frame.shape[0]:
            return
        if x_offset <= 0 or x_offset + img.shape[1] >= self.current_frame.shape[1]:
            return
        self.current_frame[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]] = img

    def renderBoxes(self, boundingBoxes, boxState, frame, matchFaces=None):
        self.current_frame = frame
        frame_size = np.asarray(frame.shape)[0:2]

        for boxCorners in boundingBoxes:
            y = int(np.maximum(boxCorners[1]-32/2, 0))
            x = int(np.maximum(boxCorners[0]-32/2, 0))
            w = int(np.minimum(boxCorners[2]+32/2, frame_size[1]) - x)
            h = int(np.minimum(boxCorners[3]+32/2, frame_size[0]) - y)

            self.draw_face_bounding_box(x, y, w, h, boxState)

            if boxState == FACESTATE.MATCH_FOUND:
                self.draw_text_labels(x, y, w, h,
                                          (matchFaces[0][0], matchFaces[0][1], cv2.IMREAD_COLOR),
                                          (matchFaces[1][0], matchFaces[1][1], cv2.IMREAD_COLOR),
                                          (matchFaces[2][0], matchFaces[2][1], cv2.IMREAD_COLOR))

        cv2.imshow(self.window_name, self.current_frame)
        cv2.waitKey(1) & 0xFF == ord('q')
