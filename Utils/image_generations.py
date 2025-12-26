# utils\image_generation.py

import cv2
import time
import numpy as np

from utils.global_definitions import (
    green_bgr,
    width, height, 
    small_aruco_side_length, aruco_marker_dictionary, aruco_marker_ids
)

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

def create_bitgrid_frame(bitgrid, width, height):

    cell_row = 0
    cell_col = 0

    

    for byte in bitgrid:
        for bit in byte:



def create_aruco_marker_frame():

    """
    Creates a solid color frame with ArUco markers in each corner.

    Arguments:
        None

    Returns:
        "frame": The created frame.

    """

    frame = create_frame_bgr(green_bgr, width, height)

    aruco_marker_positions = [
        (0, 0), # Top-left marker
        (width - small_aruco_side_length, 0), # Top-right marker
        (0, height - small_aruco_side_length), # Bottom-left marker
        (width - small_aruco_side_length, height - small_aruco_side_length) # Bottom-right marker
    ]

    for (x_coordinate, y_coordinate), aruco_marker_id in zip(aruco_marker_positions, aruco_marker_ids):
        marker = cv2.aruco.generateImageMarker(aruco_marker_dictionary, aruco_marker_id, small_aruco_side_length)
        marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        frame[y_coordinate:y_coordinate + small_aruco_side_length, x_coordinate:x_coordinate + small_aruco_side_length] = marker_bgr

    return frame

if __name__ == "__main__":

    display = create_frame_bgr((0, 255, 0), 1920, 1080)

    cv2.imwrite("optical-communication/utils/Test.png", display)