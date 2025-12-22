# receivers/receiver_v1.py

import cv2
import numpy as np
import time

from webcam_simulation.threaded_webcam import threaded_webcam

from utils.global_definitions import (
    blue_bgr, green_bgr, red_bgr, black_bgr, white_bgr,
    width, height
)


class receiver:

    def __init__(self, video_cap):

        self.video_cap = video_cap

        self.state = ""

        self.color = ""
        self.last_color = ""

        self.roi_coordinates = (
            width//2 - 25,
            width//2 + 25,
            height//2 - 25,
            height//2 + 25
        )
    
    def receive_message(self):
        """
        Docstring for receive_frames
        
        :param self: Description
        """

        self.state = "waiting for green frame"

        while True:
            
            if self.state == "waiting for green frame":
                if self.color != "green" and self.last_color == "green":
                    self.state = "decoding bits"

            elif self.state == "decoding bits":

                if self.color != "green":
                    self.state = "decoding message"

            elif self.state == "decoding message":


if __name__ == "__main__":

    video_path = "recordings/sender_v1.mp4"

    video_cap = threaded_webcam(video_path, False, True)

    if not video_cap.isOpened():

        print("Error: Could not open camera/video.")
        exit()

    cv2.namedWindow("Webcam Receiver", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Webcam Receiver", width, height)

    cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ROI", width, height)

    # Grab one initial frame so cap is "warmed up"
    while True:

        ret, frame = video_cap.read()

        if ret:
            break
        time.sleep(0.01)
    
    receiver_ = receiver(video_cap)
    receiver_.receive_message()