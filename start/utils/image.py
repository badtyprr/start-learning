# Image utilities

import cv2
import numpy as np

def quantized_histogram(image, bits=2):
    # Decimate to 2-bits
    image = cv2.divide(image, int(pow(2, 8-bits)))
    hist = np.zeros(pow(2,bits*3), dtype=np.uint)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            index = (image[row, col, 0] << 4) + (image[row, col, 1] << 2) \
                + (image[row, col, 2]) - 1
            hist[index] += 1
    return hist