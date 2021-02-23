import cv2
import numpy as np
from random import *

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

i = 1
r = randint(1, 10000000)
dummy_frame = np.empty((720, 1280), int)
while True:
    ret, frame = cap.read()

    # cv2.imwrite('my_face/' + str(r) + str(i) + '.jpg', frame)
    cv2.imshow('face', frame)

    print(i)
    i += 1

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
