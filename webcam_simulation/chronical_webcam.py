# webcam_simulation/chronical_webcam.py

import cv2
import time

from utils.global_definitions import (
    width, height
)

# --- No threading at all ---

class chronical_webcam:

    """
    Synchronous, zero-thread video reader. Returns frames EXACTLY in the order they are encoded.

    """

    def __init__(self, video_path, loop = False):

        """
        
        """

        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        self.loop = loop
        self.stopped = False

    def read(self):

        """
        
        """

        if self.stopped:
            return False, None

        ret, frame = self.cap.read()

        if not ret:

            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()

            else:
                self.stopped = True
                return False, None

        return True, frame

    def isOpened(self):

        """

        """

        return not self.stopped

    def release(self):

        """

        """

        self.stopped = True
        self.cap.release()


if __name__ == "__main__":

    # Video path

    video_path = "recordings/sender_v1.mp4"

    video_capture = chronical_webcam(video_path)

    if not video_capture.isOpened():
        print("\n[WARNING] Couldn't start video capture.")
        exit()

    while True:

        read_was_sucessful, frame = video_capture.read() # Tries to grab one initial frame to make sure the video capture is "warmed up"

        if read_was_sucessful:
            break

        time.sleep(0.01)

    cv2.namedWindow("sjöbo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("sjöbo", width, height)

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if ret:
            cv2.imshow("sjöbo", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    video_capture.release()
    cv2.destroyAllWindows()


