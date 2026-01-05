# receivers/receiver_v7.py

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import time
import queue
import multiprocessing

from webcam_simulation.process_webcam import VideoProcessCapture

from decoding_pipeline.shared_functions import Shared
from decoding_pipeline.pipeline import Pipeline_message

from utils.color_functions_bgr import dominant_color as dominant_color_bgr
from utils.color_functions_hcv import build_color_LUT, bitgrid, bitgrid_majority_calculator, range_calibration, \
dominant_color_hcv, bgr_to_hcv
from utils.screen_alignment import homography_from_large_markers, warp_alignment
from utils import decoding_functions
from utils.global_definitions import (
    green_bgr,
    width, height,
    aruco_detector_parameters, aruco_marker_dictionary
)


class receiver:

    def __init__(self, video_cap, shared=None):

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

        self.decoded_message = ""

        # Time
        self.interval = 0
        self.last_frame_time = 0

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
        self.is_decoding_started = False
        self.is_arucos_found = False
        self.is_colors_calibrated = False
        self.is_syncing = True
        self.is_syncing_initialized = False
        self.is_waiting_message = False

        # Matrix
        self.homography = None

        # Initialize classes or use provided shared instance
        self.shared = shared if shared is not None else Shared()
        
    
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
            
            self.homography, self.src_pts = homography_from_large_markers(corners, ids, width//4, height//4)

            self.warped_start_x_roi = width//8 - 25
            self.warped_start_y_roi = height//8 - 25
            self.warped_end_x_roi = width//8 + 25
            self.warped_end_y_roi = height//8 + 25
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
                print("[INFO] Sync complete, switching to color calibration...")
                self.which_method = "color_calibration"

    
    def color_calibration(self):
                    
                    if not self.is_colors_calibrated:

                        try:

                            #corrected_ranges = color_offset_calculation(roi)
                            corrected_ranges = range_calibration(self.warped)
                            LUT, color_names = build_color_LUT(corrected_ranges)
                            bitgrid.colors(LUT, color_names)

                            warmup_all() # Warming up numba for use
                            
                            self.is_colors_calibrated = True

                        except Exception as e:
                            print("\n[INFO] Color calibration error:", e)

                    if self.is_colors_calibrated and self.color != "yellow":
                        print("[INFO] Colors calibrated, switching to syncing...")
                        self.which_method = "syncing"

    def syncing(self):

        if self.color in ["black", "white"]: # If we're syncing:
                        
            if not self.is_syncing_initialized:
                print("\n[INFO] Trying to sync and get the interval...")
                self.is_syncing_initialized = True
            
        if self.is_syncing:
            try:
                self.interval, self.is_syncing = decoding_functions.sync_interval_detector(self.color, True) # Try to sync and get the interval

            except Exception as e:
                print("\n[WARNING] Sync error:", e)
                self.is_syncing = False
            
        elif self.color != "blue" and self.last_color == "blue":
            print(f"\n[INFO] Interval: {self.interval} s")
            print("[INFO] Syncing complete, switching to decoding...")
            self.which_method = "decoding_bits"


    def decoding_bits(self):
        """
        Decoding bits through color recognition

        Arguments:
            self

        Returns:
            None
        """

        recall = False
        end_frame = False
        add_frame = False

        if self.last_frame_time == 0:
            self.last_frame_time = time.time()

        current_time = time.time()
        frame_time = current_time - self.last_frame_time 

        if (self.interval > 0) and (frame_time >= self.interval):
            end_frame = True
            add_frame = True 
            self.last_frame_time = current_time 

        elif self.color in ["white", "black"]:
            
            # add_frame → add frame to array

            add_frame = True

        elif self.color == "orange" and self.last_color != "orange":

            recall = True

        try:
            self.warped = bgr_to_hcv(self.warped)
            frame_data = (self.warped, add_frame, end_frame) # Create a tuple with the frame data
            self.shared.push_frame(frame_data)
        except queue.Full:
            pass  # skip if queue is full

        # When done with pushing valid frames, wait for message to be decoded
        if self.color == "orange":
            print("[INFO] Orange detected, last valid frame pushed")
            self.which_method = "waiting_for_message"
        
    def waiting_for_message(self):
        if not self.is_waiting_message:
            print("\n[INFO] Waiting for message to be decoded...")
            self.is_waiting_message = True

        self.decoded_message = self.shared.pull_decoded_message()


    def default(self):
        print("[WARNING] No method pinpointed, fallback activated")


    def state(self):
        # Function for easier handling of each state of the program by using getattr \
        # to find methods dynamically by name
        # Using "()" at the end to call the method 
        # If no method found then it returns default as a safety net
        getattr(self, self.which_method, self.default)()


    def decrypt_message(self):
        """
        Docstring for receive_frames
        
        :param self: Description
        """

        while True:
            
            # Grabbing frames
            ret, frame = self.video_cap.read()
            
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
            
            if bitgrid.LUT is None:
                # Reads the dominant color inside the ROI
                self.color = dominant_color_bgr(self.roi)
            else:
                self.roi = cv2.cvtColor(self.roi, cv2.COLOR_BGR2HSV)
                self.color = dominant_color_hcv(self.roi)


            # Searches for arucos until found
            if not self.is_arucos_found:
                gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                corners, ids, rejected = self.aruco_detector.detectMarkers(gray)

                if corners and len(ids) > 0:
                    self.is_arucos_found = True

            # Initialize the methods when arucos is detected
            if not self.is_decoding_started and self.is_arucos_found:
                self.which_method = "aruco_sync"
                print("[INFO] arucos detected, aruco sync initialized...")
                self.is_decoding_started = True

            # Displaying the frames
            cv2.imshow("Webcam Receiver", self.frame)
            cv2.imshow("ROI", self.roi)
            
            if self.is_decoding_started:
                # Calls the methods 
                self.state()

            # If there is a decoded message available then return with the message
            if self.decoded_message:
                return self.decoded_message
            
            # When "q" is pressed then the code is interupted
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                print("[INFO] Receiver interupted")
                return None

            self.last_color = self.color
    

def warmup_all():
    dummy_array = np.zeros((2, 2, 8, 16, 10), dtype = np.uint8)
    bitgrid_majority_calculator(dummy_array, 5)


if __name__ == "__main__":

    multiprocessing.freeze_support()   # For Windows EXEs, harmless otherwise
    
    # --- Definitions ---

    using_webcam = False
    pipeline = Pipeline_message()

    video_path = "recordings/sender_v7.mp4"

    if using_webcam:

        videoCapture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # Resolution

        videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # White balance

        """
        videoCapture.set(cv2.CAP_PROP_AUTO_WB, 0) # Disables auto white balance
        videoCapture.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 3000)
        print(f"\n[INFO] Video capture white balance: {videoCapture.get(cv2.CAP_PROP_WB_TEMPERATURE)}")
        """

        # Exposure
        videoCapture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # Disables auto exposure
        videoCapture.set(cv2.CAP_PROP_EXPOSURE, -5) # Lower value --> darker
        print(f"\n[INFO] Video capture exposure: {videoCapture.get(cv2.CAP_PROP_EXPOSURE)}")

        # Gain
        videoCapture.set(cv2.CAP_PROP_GAIN, 0) # Disables auto gain

    else:
        videoCapture = VideoProcessCapture(video_path, False, True, core=[7]) # Initializes a video capture object with a pre-recorded video

    if not videoCapture.isOpened():
        print("\n[WARNING] Couldn't start video capture.")
        exit()

    while True:

        read_was_sucessful, frame = videoCapture.read() # Tries to grab one initial frame to make sure the video capture is "warmed up"

        if read_was_sucessful:
            break

        time.sleep(0.01)

    # Start pipeline
    pipeline.start_pipeline(core_decode_worker=[4, 3, 2], core_message_worker=[5], core_watchdog=[6])
    
    receiver_ = receiver(videoCapture, shared=pipeline.shared)
    try:
        decoded_message = receiver_.decrypt_message()
    except KeyboardInterrupt:
        print("[Main] Caught Ctrl+C — shutting down pipeline")
        pipeline.stop_pipeline()

    cv2.destroyAllWindows()
    print(f"[COMPLETE] The final message: {decoded_message}")

