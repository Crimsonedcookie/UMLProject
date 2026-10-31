from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from collect_labelled_points import collect_labeled_points
from diagnostics import (draw_debug_registration, draw_source_overlay, blank_panel_like, make_mosaic)
from perform_homography import compute_homography_with_fallback
from pipeline_config import PipelineConfig


def analyse_frame(
    frame: np.ndarray,
    frame_path: Path,
    frame_index: int,
    total_frames: int,
    reference_image: np.ndarray,
    reference_points: dict[tuple[int, str, int, int], np.ndarray],
    detector,
    dictionary,
    config: PipelineConfig,
    display_scale: float,
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[dict], str]:
    current_points, markers = collect_labeled_points(
        image=frame,
        detector=detector,
        dictionary=dictionary,
        config=config,
    )

    if len(markers) >= config.min_markers:
        stats, mode = compute_homography_with_fallback(
            reference_points=reference_points,
            current_points=current_points,
            config=config,
        )
    else:
        stats, mode = None, "insufficient-markers"

    height, width = reference_image.shape[:2]
    if stats is None:
        corrected = blank_panel_like(frame, "homography failed", mode)
        registration_debug = blank_panel_like(frame, "registration unavailable")
    else:
        corrected = cv2.warpPerspective(
            frame,
            stats["H"],
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        registration_debug = draw_debug_registration(corrected, stats)

    source_overlay = draw_source_overlay(
        frame=frame,
        markers=markers,
        points=current_points,
        stats=stats,
        mode=mode,
        frame_index=frame_index,
        total_frames=total_frames,
        frame_name=frame_path.name,
    )

    mosaic = make_mosaic(
        source_overlay=source_overlay,
        corrected=corrected,
        registration_debug=registration_debug,
        reference=reference_image,
        display_scale=display_scale,
    )
    return mosaic, corrected if stats is not None else None, stats, mode