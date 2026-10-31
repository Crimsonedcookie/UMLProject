from typing import Optional, Tuple

import cv2
import numpy as np

from pipeline_config import PipelineConfig
from shared_types import PointKey, PointMap


def common_correspondences(
    reference_points: PointMap,
    current_points: PointMap,
    only_outer: bool = False,
) -> Tuple[list[PointKey], np.ndarray, np.ndarray]:
    keys = sorted(set(reference_points).intersection(current_points))

    if only_outer:
        keys = [key for key in keys if key[1] == "outer"]

    reference = np.array([reference_points[key] for key in keys], dtype=np.float32)
    current = np.array([current_points[key] for key in keys], dtype=np.float32)

    return keys, current, reference


def estimate_and_validate_homography(
    keys: list[PointKey],
    current: np.ndarray,
    reference: np.ndarray,
    config: PipelineConfig,
) -> Optional[dict]:
    if len(keys) < 4:
        return None

    homography, mask = cv2.findHomography(
        current,
        reference,
        method=cv2.RANSAC,
        ransacReprojThreshold=config.ransac_reproj_threshold_px,
        maxIters=4000,
        confidence=0.999,
    )



    if homography is None or mask is None:
        return None

    if not np.all(np.isfinite(homography)):
        return None

    transformed = cv2.perspectiveTransform(
        current.reshape(-1, 1, 2),
        homography,
    ).reshape(-1, 2)

    errors = np.linalg.norm(transformed - reference, axis=1)
    inliers = mask.ravel().astype(bool)

    if not np.any(inliers):
        return None

    inlier_errors = errors[inliers]
    inlier_ratio = float(np.mean(inliers))
    inlier_marker_ids = {
        keys[index][0] for index, is_inlier in enumerate(inliers) if is_inlier
    }

    stats = {
        "H": homography,
        "mask": inliers,
        "keys": keys,
        "current": current,
        "reference": reference,
        "transformed": transformed,
        "errors": errors,
        "num_points": int(len(keys)),
        "num_inliers": int(np.sum(inliers)),
        "inlier_ratio": inlier_ratio,
        "inlier_marker_ids": inlier_marker_ids,
        "mean_error_px": float(np.mean(inlier_errors)),
        "median_error_px": float(np.median(inlier_errors)),
        "rms_error_px": float(np.sqrt(np.mean(inlier_errors**2))),
        "p95_error_px": float(np.percentile(inlier_errors, 95)),
        "max_error_px": float(np.max(inlier_errors)),
    }

    valid = (
        stats["inlier_ratio"] >= config.min_inlier_ratio
        and len(stats["inlier_marker_ids"]) >= config.min_inlier_markers
        and stats["median_error_px"] <= config.max_inlier_median_error_px
    )

    stats["valid"] = bool(valid)
    return stats


def compute_homography_with_fallback(
    reference_points: PointMap,
    current_points: PointMap,
    config: PipelineConfig,
) -> Tuple[Optional[dict], str]:
    # First attempt: external + canonically matched internal points.
    keys, current, reference = common_correspondences(
        reference_points,
        current_points,
        only_outer=False,
    )
    all_stats = estimate_and_validate_homography(
        keys=keys,
        current=current,
        reference=reference,
        config=config,
    )

    if all_stats is not None and all_stats["valid"]:
        return all_stats, "external+internal"

    # Safety fallback: use only 12 external points, which already provide a
    # strong three-marker planar homography.
    outer_keys, outer_current, outer_reference = common_correspondences(
        reference_points,
        current_points,
        only_outer=True,
    )
    outer_stats = estimate_and_validate_homography(
        keys=outer_keys,
        current=outer_current,
        reference=outer_reference,
        config=config,
    )

    if outer_stats is not None and outer_stats["valid"]:
        return outer_stats, "external-only-fallback"

    return None, "failed"