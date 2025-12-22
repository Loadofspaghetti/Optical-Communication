# utils/screen_recorder.py

import time
import threading
import cv2
import numpy as np
import mss
import os


class ScreenRecorder:
    def __init__(self, output, fps=30, monitor_index=1):
        self.output = output
        self.fps = fps
        self.monitor_index = monitor_index

        self._stop_event = threading.Event()
        self._thread = None

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output), exist_ok=True)

    def _record_loop(self):
        with mss.mss() as sct:
            monitor = sct.monitors[self.monitor_index]

            width = monitor["width"]
            height = monitor["height"]

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                self.output, fourcc, self.fps, (width, height)
            )

            if not writer.isOpened():
                raise RuntimeError("Failed to open video writer")

            frame_interval = 1.0 / self.fps
            next_frame_time = time.perf_counter()

            while not self._stop_event.is_set():
                frame = np.array(sct.grab(monitor))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                writer.write(frame)

                next_frame_time += frame_interval
                sleep_time = next_frame_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)

            writer.release()

    def start(self):
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._record_loop, daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()
