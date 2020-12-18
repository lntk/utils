import numpy as np
import cv2


class Dilate(object):
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def __call__(self, image):
        return transform_image(transform_function=self.dilate, image=image)

    def dilate(self, image, niter=1):
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        dilation = cv2.dilate(image, kernel, iterations=niter)
        return dilation


class Upsampling(object):
    def __init__(self, size=64):
        self.size = size

    def __call__(self, image):
        return transform_image(transform_function=self.upsampling, image=image)

    def upsampling(self, image):
        return cv2.resize(image, (self.size, self.size))


def transform_image(transform_function, image):
    image = np.array(image)
    image = transform_function(image)

    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)

    return image
