from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PipelineConfig:
    dictionary_name: str = "DICT_4X4_50"
    marker_ids: Tuple[int, ...] = (1, 2, 3)
    border_bits: int = 1

    use_internal_corners: bool = True
    refine_external_corners: bool = True
    refine_internal_corners: bool = True

    # Local ROI size is selected from the projected marker-cell size.
    roi_half_size_fraction_of_cell: float = 0.35
    min_roi_half_size_px: int = 4
    max_roi_half_size_px: int = 14

    # Local refinement safeguards.
    max_refinement_shift_fraction_of_cell: float = 0.45
    corner_subpix_max_iter: int = 60
    corner_subpix_eps: float = 1e-3
    corner_blur_sigma: float = 0.8

    # Global homography parameters.
    ransac_reproj_threshold_px: float = 0.4
    min_markers: int = 3
    min_inlier_ratio: float = 0.55
    min_inlier_markers: int = 3
    max_inlier_median_error_px: float = 1.5

    debug_every: int = 0