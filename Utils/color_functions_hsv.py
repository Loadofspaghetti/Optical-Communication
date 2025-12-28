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

    width, height, rows, columns, decode_bit_steps, dominant_color_steps,
    delta_h
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

    def colors(self, LUT, color_names):
        self.LUT = LUT
        self.color_names = color_names


    # --- Helper Methods ---


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

        if len(self.frames) == 0:
            return None

        frames = np.asarray(self.frames)
        self.frames = []

        padded_frames, self.cell_h, self.cell_w = self._pad_frames(frames)

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
        bitgrid = bitgrid_majority_calculator(merged, number_of_classes)

        black_idx = 1
        bitgrid_str = np.where(bitgrid == black_idx, "0", "1")

        return bitgrid_str
        

    def reset(self):
        self.frames = []


bitgrid = Bitgrid()

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


def dominant_color_hsv(hsv):

    LUT = bitgrid.LUT
    names = bitgrid.color_names
    
    ph, pw, _ = hsv.shape
    
    if ph > dominant_color_steps and pw > dominant_color_steps:
        hsv = hsv [::dominant_color_steps, ::dominant_color_steps, :]

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    classes = LUT[H, S, V]

    hist = np.bincount(classes.ravel(), minlength = len(names))

    names = names[int(hist.argmax())]

    if names == "red1" or names == "red2":
        return "red"
    
    else:
        return names
    

def build_color_LUT(corrected_ranges):

    """
    Build a 180 x 256 x 256 LUT mapping HSV -> class index. Class indices follow the order of corrected_ranges keys.

    Arguments:
        "corrected_ranges"

    Returns:
        "LUT"
        "color_names"

    """

    color_names = list(corrected_ranges.keys())

    LUT = np.zeros((180, 256, 256), dtype = np.uint8)

    H = np.arange(180)[:, None, None]
    S = np.arange(256)[None, :, None]
    V = np.arange(256)[None, None, :]

    H = np.broadcast_to(np.arange(180, dtype = np.uint16)[:,None,None], (180,256,256))
    S = np.broadcast_to(np.arange(256, dtype = np.uint16)[None,:,None], (180,256,256))
    V = np.broadcast_to(np.arange(256, dtype = np.uint16)[None,None,:], (180,256,256))

    for idx, (_, (lower, upper)) in enumerate(corrected_ranges.items()):

        lh, ls, lv = lower
        uh, us, uv = upper

        if lh <= uh:
            mask = (
                (H >= lh) & (H <= uh) &
                (S >= ls) & (S <= us) &
                (V >= lv) & (V <= uv)
)

        else:
            mask = (
                ((H >= lh) | (H <= uh)) &
                (S >= ls) & (S <= us) &
                (V >= lv) & (V <= uv)
)

        LUT[mask] = idx

    return LUT, color_names


def range_calibration(roi):
    
    original_hsv_ranges = { "white": ([0, 0, 150], [179, 40, 255]),
                            "black": ([0, 0, 0], [179, 255, 50]), 
                            "red": ([0, 40, 60], [10, 255, 255]),
                            "green": ([40, 40, 60], [80, 255, 255]), 
                            "blue": ([100, 40, 60], [140, 255, 255]),
                            "yellow":([20, 40 ,60],[40, 255,255]),
                            "cyan": ([80, 40, 60],[100, 255,255]),
                            "magenta":([140, 40,60],[160,255,255]), 
                            "orange": ([10, 40, 60],[20, 255, 255])}

    roi_hcv = cv2.colorChange(roi, cv2.COLOR_BGR2HSV)

    colors_to_calibrate = ["red", "green", "blue", "yellow", "cyan", "magenta", "orange"]
    stripe_width = roi.shape[1] // len(colors_to_calibrate)
    patch_width = stripe_width // 2
    start_offset = (stripe_width - patch_width) // 2

    corrected_ranges = {}

    # Keep black and white ranges unchanged
    for color in ["white", "black"]:
        corrected_ranges[color] = original_hsv_ranges[color]

    for idx, color in enumerate(colors_to_calibrate):
        x_start = idx * stripe_width + start_offset
        x_end = x_start + patch_width

        stripe_patch = roi_hcv[:, x_start:x_end]
        median_hcv = np.median(stripe_patch.reshape(-1, 3), axis=0)
        observed_hue, observed_saturation, observed_value = median_hcv

        print(f"[CALIBRATION] {color}: Observed HCV = H:{observed_hue:.1f}, C:{observed_saturation:.1f}, V:{observed_value:.1f}")

        # Keep original C & V
        lower_orig, upper_orig = original_hsv_ranges[color]
        lower_orig = np.array(lower_orig, dtype=float)
        upper_orig = np.array(upper_orig, dtype=float)

        # Apply delta_h around observed hue
        lower_h = (observed_hue - delta_h) % 180
        upper_h = (observed_hue + delta_h) % 180

        lower_corrected = np.array([lower_h, lower_orig[1], lower_orig[2]], dtype=int)
        upper_corrected = np.array([upper_h, upper_orig[1], upper_orig[2]], dtype=int)

        corrected_ranges[color] = (lower_corrected, upper_corrected)

    print(f"corrected ranges: {corrected_ranges}")
    return corrected_ranges