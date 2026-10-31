from typing import Tuple

import cv2
import numpy as np

from shared_types import PointKey


def marker_cell_matrix(
    dictionary,
    marker_id: int,
    border_bits: int,
) -> np.ndarray:
    """
    Generate the exact binary cell matrix for a known marker ID.

    For a 4x4 dictionary with border_bits=1, the matrix is 6x6:
        1 border + 4 payload cells + 1 border.
    """
    total_cells = int(dictionary.markerSize) + 2 * int(border_bits)
    marker_image = np.zeros((total_cells, total_cells), dtype=np.uint8)

    cv2.aruco.generateImageMarker(
        dictionary,
        int(marker_id),
        total_cells,
        marker_image,
        int(border_bits),
    )

    return (marker_image > 127).astype(np.uint8)


def is_true_binary_corner(block_2x2: np.ndarray) -> bool:
    """
    Return True when a 2x2 cell neighbourhood creates a real 2-D image corner.

    Included:
    - one white / three black cells;
    - three white / one black cells;
    - diagonal checkerboard junctions.

    Excluded:
    - uniform regions;
    - simple straight horizontal or vertical edges.
    """
    white_count = int(np.sum(block_2x2))

    if white_count in (1, 3):
        return True

    if white_count == 2:
        diagonal = (
            block_2x2[0, 0] == block_2x2[1, 1]
            and block_2x2[0, 1] == block_2x2[1, 0]
            and block_2x2[0, 0] != block_2x2[0, 1]
        )
        return bool(diagonal)

    return False


def canonical_internal_corner_specs(
    dictionary,
    marker_id: int,
    border_bits: int = 1,
) -> list[Tuple[PointKey, np.ndarray]]:
    """
    Return the internal corners in a fixed canonical order.

    Each point is identified by:
        (marker_id, "inner", grid_row, grid_column)

    The normalized point coordinate is:
        x = grid_column / total_cells
        y = grid_row / total_cells
    """
    cells = marker_cell_matrix(dictionary, marker_id, border_bits)
    total_cells = cells.shape[0]

    specs: list[Tuple[PointKey, np.ndarray]] = []

    for grid_row in range(1, total_cells):
        for grid_col in range(1, total_cells):
            neighbourhood = cells[
                grid_row - 1 : grid_row + 1,
                grid_col - 1 : grid_col + 1,
            ]

            if is_true_binary_corner(neighbourhood):
                key: PointKey = (
                    int(marker_id),
                    "inner",
                    int(grid_row),
                    int(grid_col),
                )
                normalized = np.array(
                    [grid_col / total_cells, grid_row / total_cells],
                    dtype=np.float32,
                )
                specs.append((key, normalized))

    return specs


def project_normalized_marker_points(
    normalized_points: np.ndarray,
    external_corners: np.ndarray,
) -> np.ndarray:
    ideal_external = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    local_homography = cv2.getPerspectiveTransform(
        ideal_external,
        external_corners.astype(np.float32),
    )

    projected = cv2.perspectiveTransform(
        normalized_points.reshape(-1, 1, 2).astype(np.float32),
        local_homography,
    )

    return projected.reshape(-1, 2)


def estimate_marker_cell_size_px(
    external_corners: np.ndarray,
    total_cells: int,
) -> float:
    top = np.linalg.norm(external_corners[1] - external_corners[0])
    right = np.linalg.norm(external_corners[2] - external_corners[1])
    bottom = np.linalg.norm(external_corners[2] - external_corners[3])
    left = np.linalg.norm(external_corners[3] - external_corners[0])

    marker_side_px = float(np.mean([top, right, bottom, left]))
    return marker_side_px / float(total_cells)
