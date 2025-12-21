# senders\sender_v1.py

import cv2
import time

from utils.image_generations import create_frame_bgr

from utils.global_definitions import (
    red_bgr, green_bgr, blue_bgr, white_bgr, black_bgr,
    width, height
)

start_time = 0
encode_time = 0
end_time = 0

message = "HELLO"

frame = None

def sender():
    """
    Runs a series of shifting colors to be intepreted to bits
    by using openCV and imshow

    Arguments:
        None

    Returns:
        None
    """

    start_time = time.time()
    while time.time() - start_time < 0.5:

        frame = create_frame_bgr(green_bgr, width, height)
        cv2.imshow(window, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return



if __name__ == "__main__":

    window = "sender"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) # Sets the window to fullscreen

    # Runs the sender for receiever version 1
    sender()