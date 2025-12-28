# utils\global_definitions.py

import cv2

# --- BGR definitions ---

red_bgr = (0, 0, 255)
green_bgr = (0, 255, 0)
blue_bgr = (255, 0, 0)
yellow_bgr = (0, 255, 255)
black_bgr = (0, 0, 0)
white_bgr = (255, 255, 255)
gray_bgr = (128, 128, 128)
orange_bgr = (0, 140, 255)
cyan_bgr = (255, 255, 0)
magenta_bgr = (255, 0, 255)

# --- HSV definitions ---

red_lower_hsv_limit_1 = (0, 100, 100)
red_upper_hsv_limit_1 = (10, 255, 255)
red_lower_hsv_limit_2 = (160, 100, 100)
red_upper_hsv_limit_2 = (179, 255, 255)

white_lower_hsv_limit = (0, 0, 200)
white_upper_hsv_limit = (180, 50, 255)

black_lower_hsv_limit = (0, 0, 0)
black_upper_hsv_limit = (180, 255, 50)

green_lower_hsv_limit = (40, 50, 50)
green_upper_hsv_limit = (80, 255, 255)

blue_lower_hsv_limit = (100, 150, 0)
blue_upper_hsv_limit = (140, 255, 255)

# --- HCV definitions ---

red_lower_hcv_limit_1 = (0, 100, 100)
red_upper_hcv_limit_1 = (10, 255, 255)
red_lower_hcv_limit_2 = (160, 100, 100)
red_upper_hcv_limit_2 = (179, 255, 255)

white_lower_hcv_limit = (0, 0, 200)
white_upper_hcv_limit = (180, 50, 255)

black_lower_hcv_limit = (0, 0, 0)
black_upper_hcv_limit = (180, 255, 50)

green_lower_hcv_limit = (40, 50, 50)
green_upper_hcv_limit = (80, 255, 255)

blue_lower_hcv_limit = (100, 150, 0)
blue_upper_hcv_limit = (140, 255, 255)

delta_h = 5
delta_c = 15
delta_v = 15

# --- Color maps ---
"""
Maps the color to its relative bits
"""

color_map_1bit = [
    black_bgr,      # 0b0 = Black
    white_bgr       # 0b1 = White
]

color_map_2bit = [
    black_bgr,      # 0b00 = Black
    white_bgr,      # 0b01 = White
    blue_bgr,       # 0b10 = Blue 
    green_bgr       # 0b11 = Green
]

color_map_3bit = [
    black_bgr,      # 0b000 = Black
    white_bgr,      # 0b001 = White
    red_bgr,        # 0b010 = Red
    green_bgr,      # 0b011 = Green
    blue_bgr,       # 0b100 = Blue
    yellow_bgr,     # 0b101 = Yellow
    cyan_bgr,       # 0b110 = Cyan
    magenta_bgr     # 0b111 = Magenta
]

# --- idx to bits maps ---
"""
Bits per cell by color
"""
idx_to_1bit = {
    1: 0,  # black
    0: 1,  # white
}

idx_to_2bit = {
    1: 0b00,  # black
    0: 0b01,  # white
    5: 0b10,  # blue
    4: 0b11,  # green
}

idx_to_3bit = {
    0: 0b001,  # white
    1: 0b000,  # black
    2: 0b010,  # red1
    3: 0b011,  # green
    4: 0b100,  # blue
    5: 0b101,  # yellow
    6: 0b110,  # cyan
    7: 0b111,  # magenta
}

# --- Screen definitions ---

width = 1920
height = 1080

margin = 15

rows = 8
columns = 8

bits_per_cell = 1
number_of_colors = 2 ** bits_per_cell 

decode_bit_steps = 4
dominant_color_steps = 4

# --- ArUco marker definitions ---

aruco_marker_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_detector_parameters = cv2.aruco.DetectorParameters()

small_aruco_side_length = height//5
large_aruco_side_length_without_margin = height
aruco_marker_ids = [0, 1, 3, 2]

aruco_marker_frame_duration = 1