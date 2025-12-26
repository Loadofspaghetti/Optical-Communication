# senders\sender_v3.py

import cv2
import time
import numpy as np

from utils.encoding_functions import Encode
from utils.image_generations import Create_frame
from utils.screen_recorder import ScreenRecorder

from utils.global_definitions import (
    red_bgr, green_bgr, blue_bgr, white_bgr, black_bgr,
    gray_bgr, color_map_1bit, bits_per_cell,
    width, height, margin, rows, columns
)


class sender:

    def __init__(self, message):
        
        # Timers
        self.start_time = 0.5
        self.bit_time = 0.3
        self.end_time = 0.5
        self.timer = 0

        # Arrays
        self.bitgrid = []

        # Message
        self.message = message

        self.color_map = color_map_1bit
        self.bits_per_cell = bits_per_cell
        self.frame = None
        self.interupted = False

        # Dimensions
        self.width = width
        self.height = height
        self.rows = rows
        self.columns = columns

        # Initialize classes
        self.create_frame = Create_frame(self.width, self.height, self.color_map)
        self.encode = Encode(self.rows, self.columns)

        self.aruco_frame = self.create_frame.aruco_marker()


    # --- Helper method ---

    def phase(self, which_phase):
        getattr(self, which_phase)()

    def frame_with_margin(self, margin=margin):
        """
        Places `frame` inside a full-screen background with a fixed pixel margin.
        If the frame is too big to fit with the margin, it will be scaled down
        while preserving aspect ratio.
        """
        fh, fw = self.frame.shape[:2]

        # Compute available area inside the margin
        available_w = width - 2 * margin
        available_h = height - 2 * margin

        # Determine if scaling is needed
        scale = min(1.0, min(available_w / fw, available_h / fh))  # max scale = 1 (no upscaling)

        new_w = int(fw * scale)
        new_h = int(fh * scale)

        # Resize frame if needed
        if scale < 1.0:
            self.frame = cv2.resize(self.frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Create background
        background = np.full((height, width, 3), gray_bgr, dtype=np.uint8)

        # Compute top-left corner to center the frame
        x0 = (width - new_w) // 2
        y0 = (height - new_h) // 2

        # Place the frame
        background[y0:y0+new_h, x0:x0+new_w] = self.frame

        self.frame = background


    def bit_frames(self, bgr, duration, width=width, height=height):
        """
        Gets the bgr and then keeps the frame the same for a period of time
        
        Arguments:
            bgr (tuple): An array to create the image with
            width (int): An int for the width
            height (int): An int for the height

        Returns:
            None
        """

        self.frame = self.create_frame.bgr(bgr, width, height)
        self.frame_with_margin()

        self.timer = time.time()
        while time.time() - self.timer < duration:
            cv2.imshow(window, self.frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.interupted = True
                return
            
            time.sleep(0.001)
            

    def bitbrid_frames(self, duration, width=width, height=height):
        """
        Creates a bitgrid frame and keeps the frame the same for a period of time
        
        Arguments:
            bgr (tuple): An array to create the image with
            width (int): An int for the width
            height (int): An int for the height

        Returns:
            None
        """

        self.frame = self.create_frame.bitgrid(self.bitgrid, width, height)
        self.frame_with_margin()

        self.timer = time.time()
        while time.time() - self.timer < duration:
            cv2.imshow(window, self.frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.interupted = True
                return
            
            time.sleep(0.001)
            

    # --- Phases ---

    def aruco_phase(self):

        # An aruco frame to both help the receiver to warp the image correctly
        # and to signal the beginning of the message

        self.frame = self.aruco_frame
        self.frame_with_margin()

        self.timer = time.time()
        while time.time() - self.timer < self.start_time:
            cv2.imshow(window, self.frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                print("[INFO] Interupted")
                return

            time.sleep(0.001)


    def green_phase(self):

        # A green frame to signal the end of the message

        self.frame = self.create_frame.bgr(green_bgr)
        self.frame_with_margin()

        self.timer = time.time()
        while time.time() - self.timer < self.start_time:
            cv2.imshow(window, self.frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                print("[INFO] Interupted")
                return

            time.sleep(0.001)
        

    def bit_phase(self):

        frame_bit_arrays = self.encode.message_to_bit_arrays(self.message, self.bits_per_cell) # Converts the message to frame bit arrays
        blue_frame = self.create_frame.bgr(blue_bgr)

        encoded_frames = []

        for frame_bit_array in frame_bit_arrays: # For each frame bit array:
            encoded_frame = self.create_frame.bitgrid(frame_bit_array) # Encoded the frame
            encoded_frames.append(encoded_frame) # Add the encoded frame to the list of encoded frames

        # The actual bits
        for frame in encoded_frames: # For each frame:

            frame_start_time = time.time() # Records the start time for the current frame
            while time.time() - frame_start_time < self.bit_time: # While the frame duration limit hasn't been reached:
                cv2.imshow(window, frame) # Display the current frame in the window

                if cv2.waitKey(1) & 0xFF == ord("q"): # If "Q" is pressed:
                    cv2.destroyAllWindows
                    return # Exit the function
                time.sleep(0.001) # Small sleep to prevent high CPU usage

            blue_frame_time = time.time()
            while time.time() - blue_frame_time < self.bit_time:
                cv2.imshow(window, blue_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"): # If "Q" is pressed:
                    cv2.destroyAllWindows
                    return # Exit the function
                time.sleep(0.001) # Small sleep to prevent high CPU usage


    # --- Main method ---

    def encrypted_message(self):
        """
        Runs a series of shifting colors to be intepreted to bits
        by using openCV and imshow
        """

        # Aruco sync frame
        self.phase("aruco_phase")

        # Bit phase
        self.phase("bit_phase")

        # Green end frame
        self.phase("green_phase") 



if __name__ == "__main__":

    recorder = ScreenRecorder("recordings/sender_v3.mp4", 30)
    #recorder.start()

    window = "sender"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) # Sets the window to fullscreen

    # Runs the sender for receiever version 1
    sender_ = sender("HELLO, THIS IS A MESSAGE!")
    sender_.encrypted_message()

    #recorder.stop()