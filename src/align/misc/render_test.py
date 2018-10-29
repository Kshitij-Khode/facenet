from ui.render import Renderer
from ui.render import FACESTATE
import cv2

renderer = Renderer()

# for testing face detect remove test code
cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

print(cv2.__version__)

while True:
    success, img = renderer.read_camera_frame()
    # if success: pass img to facedetect remove below test code
    if success:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        count = 0
        # Draw a custom edge based bounding box around the faces
        for (x, y, w, h) in [(100, 100, 100, 100)]:
            if count == 0:
                renderer.draw_face_bounding_box(x, y, w, h, FACESTATE.MATCH_FOUND)
                tuple_first_match = ("Ojas Sawant, 90%", cv2.imread("oj1.PNG"), cv2.IMREAD_COLOR)
                tuple_second_match = ("Xyz, 60%", cv2.imread("oj2.PNG"), cv2.IMREAD_COLOR)
                tuple_third_match = ("Abc, 50%", cv2.imread("oj3.PNG"), cv2.IMREAD_COLOR)
                renderer.draw_text_labels(x,y,w,h, tuple_first_match, tuple_second_match, tuple_third_match)
            elif count % 2 == 0:
                renderer.draw_face_bounding_box(x, y, w, h, FACESTATE.DETECTED)
            else:
                renderer.draw_face_bounding_box(x, y, w, h, FACESTATE.NO_MATCH_FOUND)
            count += 1

    renderer.render()
    if renderer.check_for_exit():
        break

renderer.close()
