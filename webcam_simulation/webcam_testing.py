
import cv2
import time

if __name__ == "__main__":

    width = 640
    height = 480

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

    while True:

        read_was_sucessful, frame = videoCapture.read() # Tries to grab one initial frame to make sure the video capture is "warmed up"

        if read_was_sucessful:
            break

        time.sleep(0.01)

    previous_time = time.time()
    frame_count = 0

    while True:

        # --- Main loop ---
        
        # Grabbing frames
        ret, frame = videoCapture.read()
        
        if not ret:
            continue

        # --- Loops per second ---

        frame_count += 1

        current_time = time.time()

        if current_time - previous_time >= 1.0:
            print(f"[INFO] Loops per second: {frame_count}")
            frame_count = 0
            previous_time = current_time

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
