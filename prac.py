from PIL import Image
from PIL import ImageEnhance
import cv2
from contrast_increaser import *

img = Image.open('mouse/opened_mouse/197.jpg')

img.show()
enhancer = ImageEnhance.Brightness(img)
enhancer.enhance(1.33).show()
enhancer.enhance(1.66).show()
enhancer.enhance(2).show()


# img1 = cv2.imread('mouse/opened_mouse/197.jpg')
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# img2 = ct_increase1(img1)


# tmp = []
# for i in range(len(img1)):
#     tmp += list(img1[i])
#     tmp += list(img2[i])
    
# print(tmp)
