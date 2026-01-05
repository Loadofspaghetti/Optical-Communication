# decoding_pipeline/watchdog.py

import time

def watchdog(
        last_decode_timestamp, 
        stop_event, 
        watchdog_on=False, 
        stall_threshold=1.0):
    """
    Watchdog process to detect decoding pipeline stalls.
    """
    if watchdog_on:
        print("[WATCHDOG] Watchdog started.", flush=True)
    else:
        print("[WATCHDOG] Watchdog disabled.", flush=True)
    try:
        while not stop_event.is_set() and watchdog_on:
            try:
                time_since_last_decode = time.time() - last_decode_timestamp.value
                if time_since_last_decode > stall_threshold:
                    print(f"[WARNING] Decode stalled! {time_since_last_decode:.2f}s since last frame.", flush=True)
            except Exception as e:
                print(f"[ERROR] Watchdog exception: {e}", flush=True)
            time.sleep(0.2)
    finally:
        print("[WATCHDOG] Watchdog exiting.", flush=True)

