# senders\sender_v1.py

import cv2
import time

from utils.image_generations import create_frame_bgr
from utils.screen_recorder import ScreenRecorder

from utils.global_definitions import (
    red_bgr, green_bgr, blue_bgr, white_bgr, black_bgr,
    width, height
)

start_time = 0.5
bit_time = 0.3
end_time = 0.5

timer = 0

message = "HELLO"

frame = None
interupted = False


def bit_frames(bgr, duration, width=width, height=height):
    """
    Gets the bgr and then keeps the frame the same for a period of time
    
    Arguments:
        bgr (tuple): An array to create the image with
        width (int): An int for the width
        height (int): An int for the height

    Returns:
        None
    """

    global interupted

    timer = time.time()
    while time.time() - timer < duration:

        frame = create_frame_bgr(bgr, width, height)
        cv2.imshow(window, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            interupted = True
            return
        

def sender():
    """
    Runs a series of shifting colors to be intepreted to bits
    by using openCV and imshow

    Arguments:
        None

    Returns:
        None
    """


    # A green start frame to signal the beginning of the sender

    timer = time.time()
    while time.time() - timer < start_time:

        frame = create_frame_bgr(green_bgr, width, height)
        cv2.imshow(window, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            print("[INFO] Interupted")
            return
        
    
    # The actual bits
        
    for character in message:

        bits = format(ord(character), "08b")

        for bit in bits:
            
            # The actual bits
            if bit == "1":
                bit_frames(white_bgr, bit_time)
            else:
                bit_frames(black_bgr, bit_time)

            if interupted:
                cv2.destroyAllWindows
                print("[INFO] Interupted")
                return
            
            # Sync frame in between each bit
            bit_frames(blue_bgr, bit_time)

            if interupted:
                cv2.destroyAllWindows
                print("[INFO] Interupted")
                return
            
        # Red frame to indicate the end of the character
        bit_frames(red_bgr, bit_time)

        if interupted:
            cv2.destroyAllWindows
            print("[INFO] Interupted")
            return

    
    # A green frame at the end to signal the end of sender
    
    timer = time.time()
    while time.time() - timer < end_time:

        frame = create_frame_bgr(green_bgr, width, height)
        cv2.imshow(window, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            print("[INFO] Interupted")
            return



if __name__ == "__main__":

    recorder = ScreenRecorder("recordings/sender_v1.mp4", 30)
    recorder.start()

    window = "sender"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) # Sets the window to fullscreen

    # Runs the sender for receiever version 1
    sender()

    recorder.stop()