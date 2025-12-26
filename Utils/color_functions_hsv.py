# utils/color_functions.py

import cv2
import numpy as np

from numba import njit, prange
from collections import Counter

from utils.global_definitions import (
    red_lower_hsv_limit_1, red_upper_hsv_limit_1,
    red_lower_hsv_limit_2, red_upper_hsv_limit_2,
    white_lower_hsv_limit, white_upper_hsv_limit,
    black_lower_hsv_limit, black_upper_hsv_limit,
    green_lower_hsv_limit, green_upper_hsv_limit,
    blue_lower_hsv_limit, blue_upper_hsv_limit,

    width, height, rows, columns, decode_bit_steps
)

class Bitgrid:

    def __init__(self, width=width, height=height, rows=rows, columns=columns):

        self.width = width
        self.height = height
        self.rows = rows
        self.columns = columns
        self.decode_bit_steps = decode_bit_steps

        # Frame array
        self.frames = []

        self.LUT = None
        self.color_names = None


    # --- Helper Methods ---

    @njit(parallel = True)
    def bitgrid_majority_calculator(patch_class_array, number_of_classes):

        """
        Aggregates patches of class labels for each cell in a grid and assigns the cell its most frequently occurring label.

        Arguments:
            "patch_class_array": A 5-dimensional array (tensor).
            "number_of_classes": The total number of distinct class labels that "patch_class_array" can contain.
        
        Returns:
            "out": A grid of majority labels.

        """

        number_of_frames, number_of_grid_rows, patch_heights, number_of_grid_columns, patch_widths = patch_class_array.shape # Extracts the five dimensions of "patch_class_array"

        out = np.empty((number_of_grid_rows, number_of_grid_columns), dtype = np.int32) # Creates an empty integer array to store the majority class for each grid coordinate

        for row in prange(number_of_grid_rows): # For each row:

            for column in range(number_of_grid_columns): # For each column:

                counts = np.zeros(number_of_classes, dtype = np.int32) # Create a histogram array to accumulate how often each class appears in the cell region

                for sample in range(number_of_frames): # For each frame:

                    for patch_height in range(patch_heights): # Loop over the patch height:

                        for patch_width in range(patch_widths): # Loop over the patch width:

                            class_id = patch_class_array[sample, row, patch_height, column, patch_width] # Extract the class ID at these coordinates

                            if 0 <= class_id < number_of_classes: # If the class ID number is within the given class ID range
                                counts[class_id] += 1 # Increase the counter for that class ID

                majority_class_id = 0 # Initialize the majority class id as 0
                majority_class_count = counts[majority_class_id] # Initializes the majority class value

                for class_id_index in range(1, number_of_classes): # Argmax to find the class with the highest frequency

                    if counts[class_id_index] > majority_class_count:
                        majority_class_id = class_id_index
                        majority_class_count = counts[class_id_index]

                out[row, column] = majority_class_id # Assign the majority class to the output grid

        return out
    

    def _pad_frames(self, frames):

        
        """
        
        """

        _, H, W, _ = frames.shape

        cell_h = int(np.ceil(H / self.rows))
        cell_w = int(np.ceil(W / self.columns))

        padded_H = cell_h * self.rows
        padded_W = cell_w * self.columns

        pad_bottom = padded_H - H
        pad_right = padded_W - W

        padded_frames = np.pad(frames, ((0, 0), (0, pad_bottom), (0, pad_right), (0, 0)), mode = "edge")

        return padded_frames, cell_h, cell_w
    

    def add_frame(self, frame):
        self.frames.append(frame)
  

    def decode_bits(self):

        if len(self.hsv_frames) == 0:
            return None

        hsv_frames = np.asarray(self.hsv_frames)
        self.hsv_frames = []

        padded_frames, self.cell_h, self.cell_w = self._pad_frames(hsv_frames)

        N, _, _, _ = padded_frames.shape

        cells = padded_frames.reshape(N, self.rows, self.cell_h, self.columns, self.cell_w, 3)

        patch_h = max(self.cell_h // 2, 1)
        patch_w = max(self.cell_w // 2, 1)

        h0 = (self.cell_h - patch_h) // 2
        h1 = h0 + patch_h
        w0 = (self.cell_w - patch_w) // 2
        w1 = w0 + patch_w

        sampled_cells = cells[:, :, h0:h1, :, w0:w1, :]

        if patch_h > self.decode_bit_steps and patch_w > self.decode_bit_steps:
            sampled_cells = sampled_cells[:, :, ::self.decode_bit_steps, :, ::self.decode_bit_steps, :]

        Hc = sampled_cells[..., 0].astype(np.uint16)
        Sc = sampled_cells[..., 1].astype(np.uint16)
        Vc = sampled_cells[..., 2].astype(np.uint16)

        classes = self.LUT[Hc, Sc, Vc]

        merged = classes

        number_of_classes = int(self.LUT.max()) + 1
        bitgrid = self.bitgrid_majority_calculator(merged, number_of_classes)

        black_idx = 1
        bitgrid_str = np.where(bitgrid == black_idx, "0", "1")

        return bitgrid_str
        


    def reset(self):
        self.frames = []

