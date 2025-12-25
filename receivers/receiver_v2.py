# receivers/receiver_v2.py

import cv2
import numpy as np
import time

from webcam_simulation.threaded_webcam import threaded_webcam
from utils.color_functions import dominant_color, average
from utils.screen_alignment import homography_from_small_arucos, warp_alignment

from utils.global_definitions import (
    blue_bgr, green_bgr, red_bgr, black_bgr, white_bgr,
    width, height,
    aruco_detector_parameters, aruco_marker_dictionary
)


class receiver:

    def __init__(self, video_cap):

        # Video capture (Grabbing frames)
        self.video_cap = video_cap
        self.frame = None
        self.warped = None
        self.roi = None

        # Strings
        self.which_method = ""

        self.color = ""
        self.last_color = ""

        self.bits = ""
        self.byte = ""

        self.message = ""
        self.decoded_message = ""

        # Coordinates
        self.frame_start_x_roi = width//2 - 25
        self.frame_end_x_roi = width//2 + 25
        self.frame_start_y_roi = height//2 - 25
        self.frame_end_y_roi = height//2 + 25

        self.warped_start_x_roi = 0
        self.warped_end_x_roi = 0
        self.warped_start_y_roi = 0
        self.warped_end_y_roi = 0

        self.src_pts = None

        # ArUco setup (match sender) 
        self.aruco_detector = cv2.aruco.ArucoDetector(aruco_marker_dictionary, aruco_detector_parameters)
        self.non_aruco_count = 0

        # Booleans
        self.decoding_started = False
        self.arucos_found = False

        # Matrix
        self.homography = None
        
    
    def aruco_sync(self):
        """
        Waiting for green frames is to be able to sync correctly 
        at the start of the bits decoding by switching states when no green
        is no longer detected
        """

        # ArUco detection on the frame

        if self.homography is None:
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
            
            self.homography, self.src_pts = homography_from_small_arucos(corners, ids, width//4, height//4)

            self.warped_start_x_roi = width//4 - 25
            self.warped_start_y_roi = height//4 - 25
            self.warped_end_x_roi = width//4 + 25
            self.warped_end_y_roi = height//4 + 25
        else:
            
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = self.aruco_detector.detectMarkers(gray)

            # Wait for arucos to disappear consistently before decoding
            if ids is None or len(ids) == 0:
                self.non_aruco_count += 1
            else:
                self.non_aruco_count = 0

            # Require a few non-green frames to avoid noise
            if self.non_aruco_count >= 3:
                print("[INFO] Sync complete, switching to decoding bits...")
                self.which_method = "decoding_bits"


    def decoding_bits(self):
        """
        Decoding bits through color recognition

        Arguments:
            self

        Returns:
            None
        """

        # When the decoding is complete switch to decoding message
        if self.color == "green":
            print("[INFO] Green detected, message complete")
            self.decoded_message = self.message

        # Grabs the average color
        if self.color == "blue" and self.last_color != "blue":

            average_color = average.majority()
            assert average_color == "white" or average_color == "black", \
                "No black or white found, out of sync..."
                        
            if average_color == "white":
                self.bits += "1"

            elif average_color == "black":
                self.bits += "0"

        # Puts the black/white frames inside ann array
        elif self.color == "black" or self.color == "white":
            average.add_frame(self.roi)

        # End-of-character marker
        elif self.color == "red" and self.last_color != "red":
            
            # Decode only FULL bytes
            while len(self.bits) >= 8:
                self.byte = self.bits[:8]
                self.bits = self.bits[8:]

                try:
                    ch = chr(int(self.byte, 2))

                except ValueError:
                    ch = '?'

                self.message += ch
                print(f"Received char: {ch}")

            if 0 < len(self.bits) < 8:
                print(f"[WARNING] Dropping incomplete byte: {self.bits}")
                self.bits = ""

            self.bits = ""


    def default(self):
        print("[WARNING] No method pinpointed, fallback activated")


    def state(self):
        # Function for easier handling of each state of the program by using getattr \
        # to find methods dynamically by name
        # Using "()" at the end to call the method 
        # If no method found then it returns default as a safefty net
        getattr(self, self.which_method, self.default)()


    def decrypt_message(self):
        """
        Docstring for receive_frames
        
        :param self: Description
        """

        while True:
            
            # Grabbing frames
            _, frame = self.video_cap.read()
            
            if not ret:
                continue

            self.frame = frame

            if self.src_pts is not None:
                self.warped = warp_alignment(self.frame, self.homography, width//4, height//4)
                cv2.polylines(self.frame, [self.src_pts.astype(np.int32)], True, (green_bgr), 2)
                
                # Extract the ROI from the warped frame
                self.roi = self.warped[self.warped_start_y_roi:self.warped_end_y_roi, self.warped_start_x_roi:self.warped_end_x_roi] 
            else:
                # Extract the ROI from the frame
                self.roi = self.frame[self.frame_start_y_roi:self.frame_end_y_roi, self.frame_start_x_roi:self.frame_end_x_roi] 

            # Reads the dominant color inside the ROI
            self.color = dominant_color(self.roi)

            # Searches for arucos until found
            if not self.arucos_found:
                gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                corners, ids, rejected = self.aruco_detector.detectMarkers(gray)

                if corners and len(ids) > 0:
                    self.arucos_found = True

            # Initialize the methods when arucos is detected
            if not self.decoding_started and self.arucos_found:
                self.which_method = "aruco_sync"
                print("[INFO] arucos detected, aruco sync initialized...")
                self.decoding_started = True

            # Displaying the frames
            cv2.imshow("Webcam Receiver", self.frame)
            cv2.imshow("ROI", self.roi)
            
            if self.decoding_started:
                # Calls the methods 
                self.state()

            # If there is a decoded message available then break
            if self.decoded_message:
                break
            
            # When "q" is pressed then the code is interupted
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                print("[INFO] Receiver interupted")
                return

            self.last_color = self.color
        
        return self.decoded_message


if __name__ == "__main__":

    video_path = "recordings/sender_v2.mp4"

    video_cap = threaded_webcam(video_path, False, True)

    if not video_cap.isOpened():

        print("Error: Could not open camera/video.")
        exit()

    cv2.namedWindow("Webcam Receiver", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Webcam Receiver", width, height)

    cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ROI", width//4, height//4)

    # Grab one initial frame so cap is "warmed up"
    while True:

        ret, frame = video_cap.read()

        if ret:
            break
        time.sleep(0.01)
    
    receiver_ = receiver(video_cap)
    decoded_message = receiver_.decrypt_message()

    cv2.destroyAllWindows()
    print(f"[COMPLETE] The final message: {decoded_message}")

