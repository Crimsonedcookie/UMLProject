from typing import Dict, Tuple

import numpy as np

from aruco_identification import detect_markers_by_id
from detect_internal_corners import (canonical_internal_corner_specs, project_normalized_marker_points,
                                     estimate_marker_cell_size_px, )
from pipeline_config import PipelineConfig
from shared_types import PointKey, PointMap
from sub_pixel_refinement import refine_point_locally
from utility_funcs import to_gray


def collect_labeled_points(
    image: np.ndarray,
    detector,
    dictionary,
    config: PipelineConfig,
) -> Tuple[PointMap, Dict[int, np.ndarray]]:
    gray = to_gray(image)

    markers = detect_markers_by_id(
        image=image,
        detector=detector,
        expected_ids=config.marker_ids,
    )

    points: PointMap = {}
    total_cells = int(dictionary.markerSize) + 2 * config.border_bits

    for marker_id in config.marker_ids:
        if marker_id not in markers:
            continue

        external = markers[marker_id].astype(np.float32)
        cell_size_px = estimate_marker_cell_size_px(external, total_cells)

        # OpenCV's external corner order is stable after ID decoding.
        for corner_index, corner in enumerate(external):
            key: PointKey = (
                int(marker_id),
                "outer",
                int(corner_index),
                -1,
            )

            point = corner.copy()
            if config.refine_external_corners:
                refined = refine_point_locally(
                    gray=gray,
                    initial_point=point,
                    cell_size_px=cell_size_px,
                    config=config,
                )
                if refined is not None:
                    point = refined

            points[key] = point.astype(np.float32)

        if not config.use_internal_corners:
            continue

        specs = canonical_internal_corner_specs(
            dictionary=dictionary,
            marker_id=marker_id,
            border_bits=config.border_bits,
        )

        if len(specs) == 0:
            continue

        normalized_points = np.vstack([coordinate for _key, coordinate in specs])
        projected_points = project_normalized_marker_points(
            normalized_points=normalized_points,
            external_corners=external,
        )

        for (key, _normalized), projected in zip(specs, projected_points):
            point = projected.astype(np.float32)

            if config.refine_internal_corners:
                refined = refine_point_locally(
                    gray=gray,
                    initial_point=point,
                    cell_size_px=cell_size_px,
                    config=config,
                )
                if refined is None:
                    # Exclude this key from this frame. The reference/current
                    # intersection will then avoid a false correspondence.
                    continue
                point = refined

            points[key] = point.astype(np.float32)

    return points, markers
