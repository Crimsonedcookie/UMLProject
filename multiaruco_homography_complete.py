"""
multiaruco_homography_complete.py

Complete, modular pipeline for planar wall stabilization using three known
ArUco markers (default IDs 1, 2, 3), including:

1. Detection and ID-based matching of each marker.
2. Four external corners per marker.
3. Internal black/white cell corners reconstructed from the known ArUco ID.
4. Sub-pixel local refinement.
5. Homography estimation with RANSAC.
6. Automatic fallback to external corners if internal refinement is unreliable.
7. PNG export and a CSV validation report.

Why this fixes the previous issue
---------------------------------
The internal points are NOT independently detected and then sorted with
lexsort. Each point has a permanent identity:

    (marker_id, "inner", grid_row, grid_column)

Therefore, the point (1, "inner", 3, 2) in the current frame is matched only
with the same physical grid corner in the reference frame.

Recommended command
-------------------
python multiaruco_homography_complete.py \
    --input-dir "/Users/nathanramani/PycharmProjects/UML/SineWaveTest1" \
    --output-dir "/Users/nathanramani/PycharmProjects/UML/SineWaveTest1_corrected" \
    --glob "sine_test1_frame_*.jpg" \
    --dictionary DICT_4X4_50 \
    --ids 1 2 3 \
    --reference-index 0 \
    --min-markers 3 \
    --ransac-threshold 0.4 \
    --debug-every 50

Notes
-----
- Use the dictionary that was used to print the markers. If the markers were
  generated with DICT_4X4_50, pass --dictionary DICT_4X4_50.
- Output frames are saved as PNG to avoid JPEG compression artefacts before DICe.
- All three markers must lie on the same physical wall plane for one global
  homography to be valid.
"""

from __future__ import annotations

from command_line_interface import *

if __name__ == "__main__":
    main()