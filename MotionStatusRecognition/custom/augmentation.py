import numpy as np
from PIL import Image, ImageEnhance

def to_PIL(img):
    assert(img.shape[2]==3)
    img = Image.fromarray(np.uint8(img), 'RGB')
    return img

def to_array(img):
    img = np.array(img)
    img = np.float32(img)
    return img

def norm(mean, scale):
    def f(img):
        img = img - mean
        img = img / scale
        return img
    return f

def random_brightness(img):
    factor = np.random.uniform(0.8, 1.2)
    img = ImageEnhance.Brightness(img).enhance(factor)
    return img

def random_rotation(img):
    p = np.random.uniform(0., 1.)
    if p <= 0.5:
        img = img.rotate(90)
    return img

def compose(*transforms):
    nb_transforms = len(transforms)
    def compose_transform(img):
        for iter in range(nb_transforms):
            f = transforms[iter]
            img = f(img)
        return img
    return compose_transform