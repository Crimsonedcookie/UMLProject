import math
from typing import Optional, Tuple

import cv2
import numpy as np

from pipeline_config import PipelineConfig


def crop_roi_around_point(
    gray: np.ndarray,
    point: np.ndarray,
    half_size: int,
) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
    height, width = gray.shape[:2]
    x, y = float(point[0]), float(point[1])

    x0 = max(0, int(math.floor(x - half_size)))
    y0 = max(0, int(math.floor(y - half_size)))
    x1 = min(width, int(math.ceil(x + half_size + 1)))
    y1 = min(height, int(math.ceil(y + half_size + 1)))

    roi = gray[y0:y1, x0:x1]
    if roi.size == 0 or min(roi.shape[:2]) < 5:
        return None, (x0, y0)

    return roi.copy(), (x0, y0)


def refine_point_locally(
    gray: np.ndarray,
    initial_point: np.ndarray,
    cell_size_px: float,
    config: PipelineConfig,
) -> Optional[np.ndarray]:
    half_size = int(
        round(config.roi_half_size_fraction_of_cell * float(cell_size_px))
    )
    half_size = int(
        np.clip(
            half_size,
            config.min_roi_half_size_px,
            config.max_roi_half_size_px,
        )
    )

    roi, (x0, y0) = crop_roi_around_point(
        gray=gray,
        point=initial_point,
        half_size=half_size,
    )
    if roi is None:
        return None

    refined_image = roi

    if config.corner_blur_sigma > 0:
        refined_image = cv2.GaussianBlur(
            refined_image,
            (0, 0),
            sigmaX=config.corner_blur_sigma,
        )

    local_x = float(initial_point[0]) - x0
    local_y = float(initial_point[1]) - y0

    initial_local = np.array(
        [
            [
                [
                    local_x,
                    local_y,
                ]
            ]
        ],
        dtype=np.float32,
    )

    point_local = initial_local.copy()
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        config.corner_subpix_max_iter,
        config.corner_subpix_eps,
    )

    try:
        cv2.cornerSubPix(
            refined_image,
            point_local,
            winSize=(3, 3),
            zeroZone=(-1, -1),
            criteria=criteria,
        )
    except cv2.error:
        return None

    refined_local_x, refined_local_y = point_local.reshape(2)

    refined_global = np.array(
        [refined_local_x + x0, refined_local_y + y0],
        dtype=np.float32,
    )

    shift = float(np.linalg.norm(refined_global - initial_point))
    maximum_shift = max(
        1.0,
        config.max_refinement_shift_fraction_of_cell * float(cell_size_px),
    )

    if not np.all(np.isfinite(refined_global)) or shift > maximum_shift:
        return None

    return refined_global