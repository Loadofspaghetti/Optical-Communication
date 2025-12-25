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

# --- Screen definitions ---

width = 1920
height = 1080

margin = 15

# --- ArUco marker definitions ---

aruco_marker_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_detector_parameters = cv2.aruco.DetectorParameters()

small_aruco_side_length = height//5
large_aruco_side_length_without_margin = height
aruco_marker_ids = [0, 1, 3, 2]

aruco_marker_frame_duration = 1