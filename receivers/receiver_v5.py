# receivers/receiver_v5.py

import cv2
import numpy as np
import time
import queue
import threading

from webcam_simulation.threaded_webcam import threaded_webcam
from utils.color_functions_bgr import dominant_color
from utils.color_functions_hsv import build_color_LUT, bitgrid, bitgrid_majority_calculator, range_calibration
from utils.screen_alignment import homography_from_small_arucos, warp_alignment
from utils import decoding_functions

from utils.global_definitions import (
    blue_bgr, green_bgr, red_bgr, black_bgr, white_bgr,
    width, height,
    aruco_detector_parameters, aruco_marker_dictionary
)

# Global definitions
decoded_message = ""


class Threads:

    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=100)
        self.last_queue_debug = 0
        self.decode_last_time = time.time()
        self.decoded_message = ""
        self.stop_thread = False
        self.watchdog_on = False

        # Start decoding thread
        decode_thread = threading.Thread(target=self.decoding_worker, daemon=True)
        decode_thread.start()

        # Start watchdog thread
        watch_thread = threading.Thread(target=self.watchdog, daemon=True)
        watch_thread.start()


    def decoding_worker(self):

        global decoded_message

        while not self.stop_thread or not self.frame_queue.empty():
            try:
                hsv_roi, recall, add_frame, end_frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            now = time.time()
            if now - self.last_queue_debug > 0.5:
                print(f"[DEBUG] Decode thread queue size = {self.frame_queue.qsize()}")
                self.last_queue_debug = now
            
            t0 = time.time()
            if recall:
                decoded_message = decoding_functions.decode_bitgrid_hsv(
                    hsv_roi, add_frame, recall, end_frame
                )
            else:
                decoding_functions.decode_bitgrid_hsv(
                    hsv_roi, add_frame, recall, end_frame
                )

            self.decode_last_time = time.time()  # helps watchdog to see if decode works or not


            t1 = time.time()

            # Print timing occasionally
            if t1 - self.last_queue_debug > 0.5:
                print(f"[DEBUG] Decode time: {(t1 - t0)*1000:.2f} ms")


    def watchdog(self):
        while self.watchdog_on:
            if time.time() - self.decode_last_time > 1.0:
                print("[WARNING] Decode thread is stalled or starving (no frames processed)!")
            time.sleep(0.2)

    

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

        # Matrix
        self.homography = None

        # Initialize Threads class
        self.threads = Threads()
        
    
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

        elif self.color == "green" and self.last_color != "green":

            recall = True

        try:
            self.warped = cv2.cvtColor(self.warped, cv2.COLOR_BGR2HSV)
            self.threads.frame_queue.put_nowait((self.warped.copy(), recall, add_frame, end_frame))
        except queue.Full:
            pass  # skip if queue is full

        # When the decoding is complete switch to decoding message
        if self.color == "green":
            #print("[INFO] Green detected, message complete")
            return


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

            # If there is a decoded message available then break
            if decoded_message:
                break
            
            # When "q" is pressed then the code is interupted
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                print("[INFO] Receiver interupted")
                return

            self.last_color = self.color
    

def warmup_all():
    dummy_array = np.zeros((2, 2, 8, 16, 10), dtype = np.uint8)
    bitgrid_majority_calculator(dummy_array, 5)


if __name__ == "__main__":

    video_path = "recordings/sender_v5.mp4"

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
    receiver_.decrypt_message()

    cv2.destroyAllWindows()
    print(f"[COMPLETE] The final message: {decoded_message}")

