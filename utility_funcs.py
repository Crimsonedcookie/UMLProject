import re
from pathlib import Path

import cv2
import numpy as np


def natural_sort_key(path: Path) -> list[object]:
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", path.name)
    ]


def resolve_dictionary(dictionary_name: str):
    if not hasattr(cv2.aruco, dictionary_name):
        available = sorted(
            name for name in dir(cv2.aruco) if name.startswith("DICT_")
        )
        raise ValueError(
            f"Unknown ArUco dictionary '{dictionary_name}'. "
            f"Examples: {available[:10]}"
        )

    dictionary_id = getattr(cv2.aruco, dictionary_name)
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def make_detector(dictionary):
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 5
    parameters.cornerRefinementMaxIterations = 50
    parameters.cornerRefinementMinAccuracy = 1e-3

    return cv2.aruco.ArucoDetector(dictionary, parameters)
