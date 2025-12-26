# utils/color_functions_bgr.py

import cv2

from collections import Counter

from utils.global_definitions import (
    red_lower_hsv_limit_1, red_upper_hsv_limit_1,
    red_lower_hsv_limit_2, red_upper_hsv_limit_2,
    white_lower_hsv_limit, white_upper_hsv_limit,
    black_lower_hsv_limit, black_upper_hsv_limit,
    green_lower_hsv_limit, green_upper_hsv_limit,
    blue_lower_hsv_limit, blue_upper_hsv_limit
)

class Average_calc:

    def __init__(self):
        self.frames = []

    def add_frame(self, frame):
        self.frames.append(frame)

    def majority(self):
        """
        Returns the majority (most common) color across collected frames
        """
        if not self.frames:
            return None
        
        colors = []

        for frame in self.frames:
            color = dominant_color(frame)
            colors.append(color)
        
        # Clear frames after processing
        self.frames = []
        
        # Find majority color
        majority_color = Counter(colors).most_common(1)[0][0]
        return majority_color

    def reset(self):
        self.frames = []

# Initialize class
average = Average_calc()

def dominant_color(frame):
    """
    Computes the dominant color in the given ROI using HSV color space.

    Arguments:
        roi: The region of interest (ROI) frame to analyze.

    Returns:
        The dominant color as a string (e.g., "red", "white", etc.).
    """

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    red_mask =  cv2.inRange(hsv, red_lower_hsv_limit_1, red_upper_hsv_limit_1) | \
                cv2.inRange(hsv, red_lower_hsv_limit_2, red_upper_hsv_limit_2)

    white_mask = cv2.inRange(hsv, white_lower_hsv_limit, white_upper_hsv_limit)

    black_mask = cv2.inRange(hsv, black_lower_hsv_limit, black_upper_hsv_limit)

    green_mask = cv2.inRange(hsv, green_lower_hsv_limit, green_upper_hsv_limit)

    blue_mask  = cv2.inRange(hsv, blue_lower_hsv_limit, blue_upper_hsv_limit)

    counts = {
        "red": int(cv2.countNonZero(red_mask)),
        "white": int(cv2.countNonZero(white_mask)),
        "black": int(cv2.countNonZero(black_mask)),
        "green": int(cv2.countNonZero(green_mask)),
        "blue": int(cv2.countNonZero(blue_mask))
    }

    return max(counts, key = counts.get)