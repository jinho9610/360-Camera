from PIL import Image
from PIL import ImageEnhance
import numpy as np
import cv2

def ct_increase1(img):
    img = Image.fromarray(img)

    enhancer = ImageEnhance.Brightness(img)
    return np.array(enhancer.enhance(1))


def ct_increase2(img):
    img = Image.fromarray(img)

    enhancer = ImageEnhance.Brightness(img)
    return np.array(enhancer.enhance(1.33))

def ct_increase3(img):
    img = Image.fromarray(img)

    enhancer = ImageEnhance.Brightness(img)
    return np.array(enhancer.enhance(1.66))

def ct_increase4(img):
    img = Image.fromarray(img)

    enhancer = ImageEnhance.Brightness(img)
    return np.array(enhancer.enhance(2))