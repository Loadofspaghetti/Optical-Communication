# utils/screen_alignment.py

import cv2
import numpy as np

from utils.global_definitions import (
    width, height
)




saved_corners = {0: None, 1: None, 2:None, 3:None} 

def homography_from_small_arucos(corners, marker_ids, width=width, height=height):
    """
    Computes a homography using four small ArUco markers.

    Arguments:
        corners: ArUco corners from OpenCV
        marker_ids: detected marker IDs
        width: Width of the wanted warped frame
        height: Height of the wanted warped frame

    Returns:
        H (homography): 3x3 homography matrix or None
        src_pts: Coordinates of the source points or None
    """

    global saved_corners

    H = None

    # Normalize marker_ids
    if hasattr(marker_ids, "flatten"):
        ids_flat = marker_ids.flatten()
    else:
        ids_flat = np.array(marker_ids).flatten()

    # Build ID -> corners mapping
    id_to_corners = {}
    for idx, marker_id in enumerate(ids_flat):
        id_to_corners[int(marker_id)] = corners[idx][0]

    # Persist only markers 0, 1, 2 and 3
    for marker_id in [0, 1, 2, 3]:
        if marker_id in id_to_corners:
            saved_corners[marker_id] = id_to_corners[marker_id]

    # Require both markers
    if  saved_corners[0] is None or saved_corners[1] is None or \
        saved_corners[2] is None or saved_corners[3] is None:

        return None, None

    # Select source points (image)
    # Order: TL, TR, BR, BL
    tl_marker  = saved_corners[0]
    tr_marker = saved_corners[1]
    br_marker = saved_corners[2]
    bl_marker = saved_corners[3]

    src_pts = np.array([
        tl_marker[0],   # TL
        tr_marker[1],  # TR
        br_marker[2],  # BR
        bl_marker[3]    # BL
    ], dtype=np.float32)

    # Define destination rectangle
    # (arbitrary units, consistent scale)
    if width < 5 or height < 5:
        return None, None

    dst_pts = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype=np.float32)

    # Compute homography
    H, _ = cv2.findHomography(src_pts, dst_pts)

    return H, src_pts


def warp_alignment(frame, H, dst_width, dst_height):
    """
    Calculates the homography matrix based on the saved corners of the ArUco markers.
    """
    warped = cv2.warpPerspective(frame, H, (dst_width, dst_height))
    ph, wh, _ = warped.shape
    assert ph > 0 and wh > 0, \
        "The warped frame height and width should be greater than 0"
    return warped