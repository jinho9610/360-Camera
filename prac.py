from PIL import Image
from PIL import ImageEnhance

img = Image.open('mouse/opened_mouse/197.jpg')

enhancer = ImageEnhance.Brightness(img)
enhancer.enhance(2).show()
