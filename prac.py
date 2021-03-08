import cv2

img = cv2.imread('chin.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('img', img)
cv2.waitKey(0)