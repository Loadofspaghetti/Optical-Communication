# utils\image_generation.py

import cv2
import time
import numpy as np

from utils.global_definitions import (
    green_bgr, red_bgr, blue_bgr, yellow_bgr, cyan_bgr, magenta_bgr, orange_bgr,
    color_map_1bit,
    width, height, rows, columns,
    small_aruco_side_length, aruco_marker_dictionary, aruco_marker_ids
)

class Create_frame:

    def __init__(self, width=width, height=height, color_map=color_map_1bit):
        self.width = width
        self.height = height

        self.cell_width = int(width//columns)
        self.cell_height = int(height//rows)

        self.color_map = color_map


    def bgr(self, bgr):
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

        empty_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        empty_frame[:] = [bgr]
        color_frame = empty_frame

        return color_frame


    def bitgrid(self, bitgrid):

        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        for r in range (rows):
            for c in range (columns):
                cell_value = int(bitgrid[r, c])
                color = self.color_map[cell_value] # Grabs the BGR
                
                # Cell coordinates
                x0 = c * self.cell_width
                x1 = x0 + self.cell_width
                y0 = r * self.cell_height
                y1 = y0 + self.cell_height

                cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), color, thickness=-1)

        return image


    def aruco_marker(self):

        """
        Creates a solid color frame with ArUco markers in each corner.

        Arguments:
            None

        Returns:
            "frame": The created frame.

        """

        frame = self.bgr(green_bgr)

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
    

    def color_reference(self):
    
        """
        Creates a reference frame with all key colors for the receiver to calibrate.

        Arguments: 
            None
        
        Returns:
            color_reference_frame (np.ndarray): The reference frame (BGR).

        """

        color_reference_frame = np.zeros((self.height, self.width, 3), dtype = np.uint8) # Creates a blank frame

        colors = [red_bgr, green_bgr, blue_bgr, yellow_bgr, cyan_bgr, magenta_bgr, orange_bgr]

        stripe_width = self.width // len(colors) # Divides the frame into equal vertical stripes for each color

        for stripe_index, color in enumerate(colors):

            x_start = stripe_index * stripe_width

            if stripe_index != len(colors) - 1: # If the stripe index isn't the last one:
                x_end = (stripe_index + 1) * stripe_width
            
            else: # Else (if it's the last one):
                x_end = self.width

            color_reference_frame[:, x_start:x_end] = color # Fill the entire stripe with the current color

        return color_reference_frame

create_frame = Create_frame()

if __name__ == "__main__":

    bitgrid = [
        ["10011001"],
        ["10011001"],
        ["10011001"],
        ["10011001"],
        ["10011001"],
        ["10011001"],
        ["10011001"],
        ["10011001"],
    ]

    display = create_bitgrid_frame(bitgrid, width, height)

    cv2.imwrite("optical-communication/utils/Test.png", display)