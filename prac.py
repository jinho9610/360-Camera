import cv2
import os

f = open("mouse_info.csv", "w")
f.write('state,image\n')

img_paths = os.listdir('mouse/opened_mouse')

for img_path in img_paths:
    f.write('open,')

    img = cv2.imread(os.path.join('mouse/opened_mouse', img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    tmp = []
    for i in range(len(img)):
        tmp += list(img[i])
    
    f.write('\"' + str(tmp) + '\"\n')


img_paths = os.listdir('mouse/closed_mouse')

for img_path in img_paths:
    f.write('close,')

    img = cv2.imread(os.path.join('mouse/closed_mouse', img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    tmp = []
    for i in range(len(img)):
        tmp += list(img[i])
    
    f.write('\"' + str(tmp) + '\"\n')

f.close()