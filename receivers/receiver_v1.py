# receivers/receiver_v1.py

import cv2
import numpy as np
import time

from webcam_simulation.threaded_webcam import threaded_webcam
from utils.color_functions import dominant_color

from utils.global_definitions import (
    blue_bgr, green_bgr, red_bgr, black_bgr, white_bgr,
    width, height
)


class receiver:

    def __init__(self, video_cap):

        # Video capture (Grabbing frames)
        self.video_cap = video_cap
        self.frame = None
        self.roi = None

        # Strings
        self.which_function = ""

        self.color = ""
        self.last_color = ""

        self.bits = ""
        self.byte = ""

        self.message = ""
        self.decoded_message = ""

        # Coordinates for the ROI
        self.roi_coordinates = (
            width//2 - 25,
            width//2 + 25,
            height//2 - 25,
            height//2 + 25
        )
        self.start_x_roi = width//2 - 25
        self.end_x_roi = width//2 + 25
        self.start_y_roi = height//2 - 25
        self.end_y_roi = height//2 + 25
        
    
    def green_sync(self):
        """
        Waiting for green frames is to be able to sync correctly 
        at the start of the bits decoding by switching states when no green
        is no longer detected
        """

        # No green frame detected, switching state
        if self.color != "green":
            print("[INFO] No green detected, switching to decoding bits...")
            self.which_function = "decoding_bits"

    def decoding_bits(self):
        """
        Docstring for decoding_bits
        """

        if self.color == "white" and self.last_color != "white":
            self.bits += "1"

        elif self.color == "black" and self.last_color != "black":
            self.bits += "0"

        elif self.color == "red" and self.last_color != "red":

            while len(self.bits) >= 8:
                self.byte = self.bits[:8]
                self.bits = self.bits[8:]

                try:
                    ch = chr(int(self.byte, 2))

                except:
                    ch = '?'

                self.message += ch
                print(f"Received char: {ch}")

            if 0 < len(self.bits) < 8:
                self.byte = self.bits.ljust(8, '0')

                try:
                    ch = chr(int(self.byte, 2))

                except:
                    ch = '?'

                self.message += ch
                print(f"Received char (padded): {ch}")
            print(f"[INFO] Decoded bits: {self.bits}")
            self.bits = ""
        
        # When the decoding is complete switch to decoding message
        if self.color == "green":
            print("[INFO] Green detected, message complete")
            self.decoded_message = self.message

    def default(self):
        print("[WARNING] No method pinpointed, fallback activated")


    def state(self):
        # Function for easier handling of each state of the program by using getattr \
        # to find methods dynamically by name
        # Using "()" at the end to call the method 
        # If no method found then it returns default as a safefty net
        getattr(self, self.which_function, self.default)()


    def receive_message(self):
        """
        Docstring for receive_frames
        
        :param self: Description
        """

        while True:
            
            # Grabbing frames
            _, frame = self.video_cap.read()
            self.frame = frame

            # Extract the ROI from the frame
            self.roi = self.frame[self.start_y_roi:self.end_y_roi, self.start_x_roi:self.end_x_roi] 
            cv2.rectangle (self.frame, (self.start_x_roi, self.start_y_roi), 
                          (self.end_x_roi, self.end_y_roi), (green_bgr), 2)
            
            # Reads the dominant color inside the ROI
            self.color = dominant_color(self.roi)

            # Initialize the methods when green is detected
            if not hasattr(self.receive_message, "decoding_started") and self.color == "green":
                self.which_function = "green_sync"
                print("[INFO] Green detected, green sync initialized...")
                receiver.receive_message.decoding_started = (True)

            # Displaying the frames
            cv2.imshow("Webcam Receiver", self.frame)
            cv2.imshow("ROI", self.roi)
            
            if hasattr(self.receive_message, "decoding_started"):
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

    video_path = "recordings/sender_v1.mp4"

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
    decoded_message = receiver_.receive_message()

    cv2.destroyAllWindows()
    print(f"[COMPLETE] The final message: {decoded_message}")

