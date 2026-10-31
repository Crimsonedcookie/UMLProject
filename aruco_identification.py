from typing import Dict, Iterable

import numpy as np

from utility_funcs import to_gray


def detect_markers_by_id(
    image: np.ndarray,
    detector,
    expected_ids: Iterable[int],
) -> Dict[int, np.ndarray]:
    """
    Detect the requested IDs and return their external corners.

    OpenCV returns corners in decoded marker order:
        0: top-left
        1: top-right
        2: bottom-right
        3: bottom-left

    The dictionary key is the marker ID, so IDs cannot become mixed between
    the reference and current frames.
    """
    gray = to_gray(image)
    corners, ids, _rejected = detector.detectMarkers(gray)

    detected: Dict[int, np.ndarray] = {}
    if ids is None:
        return detected

    expected_set = {int(marker_id) for marker_id in expected_ids}

    for marker_id, marker_corners in zip(ids.flatten().astype(int), corners):
        if marker_id in expected_set:
            detected[int(marker_id)] = marker_corners[0].astype(np.float32)

    return detected
