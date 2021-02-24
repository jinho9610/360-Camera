from PIL import Image
from PIL import ImageEnhance
import numpy as np
import cv2


def ct_increase(img):
    img = Image.fromarray(img)

    enhancer = ImageEnhance.Brightness(img)
    return np.array(enhancer.enhance(2))
