# utils\image_generation.py

import cv2
import time
import numpy as np

def create_frame_bgr(bgr, width, height):
    """
    Creates a bgr frame using the colors given with bgr
    and creates the frame size with the given height and width

    Arguments:
        bgr (tuple): An array that tells which color the frame should be
        width (int): An int that tells how wide the frame should be
        height (int): An int that tells which height the frame should be

    Return:
        frame (np.array): Returns an np.array that can be decoded to an image 
    """

    empty_frame = np.zeros((height, width, 3), dtype=np.uint8)

    empty_frame[:] = [bgr]
    color_frame = empty_frame

    return color_frame

if __name__ == "__main__":

    display = create_frame_bgr((0, 255, 0), 1920, 1080)

    cv2.imwrite("optical-communication/utils/Test.png", display)