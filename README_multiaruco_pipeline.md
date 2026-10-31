# Multi-ArUco Homography Pipeline

This package corrects a planar frame sequence using three printed ArUco markers
with IDs `1`, `2`, and `3`.

The final SineWaveTest1 configuration uses:

- ArUco dictionary `DICT_4X4_50`.
- External marker corners.
- Canonical internal black/white cell corners reconstructed from each known marker ID.
- Stable internal-point labels: `(marker_id, "inner", grid_row, grid_column)`.
- Local OpenCV `cornerSubPix` refinement.
- RANSAC homography with a `0.4 px` reprojection threshold.
- PNG output frames and a CSV validation report.

The old Fabio upscaling/resampling path was removed because it increased the
measured registration error on these frames.

## Files

- `multiaruco_homography_complete.py`: batch correction script.
- `live_verify_homography.py`: interactive visual verification tool.
- `README_multiaruco_pipeline.md`: this guide.

## Installation

```bash
pip install numpy opencv-contrib-python
```

## Batch Correction

Example command:

```bash
python multiaruco_homography_complete.py \
  --input-dir "C:/path/to/SineWaveTest1" \
  --output-dir "C:/path/to/SineWaveTest1_corrected_lowerror" \
  --glob "sine_test1_frame_*.jpg" \
  --dictionary DICT_4X4_50 \
  --ids 1 2 3 \
  --reference-index 0 \
  --min-markers 3 \
  --ransac-threshold 0.4 \
  --debug-every 50
```

On this machine, the exact command used was:

```powershell
& "C:\Users\Francesco_Gasperini\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" "C:\Users\Francesco_Gasperini\OneDrive - UMass Lowell\Desktop\Nathan zip\multiaruco_homography_complete.py" --input-dir "C:\Users\Francesco_Gasperini\OneDrive - UMass Lowell\Desktop\SineWaveTest1" --output-dir "C:\Users\Francesco_Gasperini\OneDrive - UMass Lowell\Desktop\SineWaveTest1_corrected_lowerror" --glob "sine_test1_frame_*.jpg" --dictionary DICT_4X4_50 --ids 1 2 3 --reference-index 0 --min-markers 3 --ransac-threshold 0.4 --debug-every 50
```

## Live Verification

```bash
python live_verify_homography.py \
  --input-dir "C:/path/to/SineWaveTest1" \
  --dictionary DICT_4X4_50 \
  --ids 1 2 3 \
  --ransac-threshold 0.4
```

Controls:

- `Space`: pause / resume.
- `N` or Right arrow: next frame when paused.
- `P` or Left arrow: previous frame when paused.
- `S`: save the current verification mosaic.
- `Q` or `Esc`: quit.

## Optional Test Modes

External marker corners only:

```bash
python multiaruco_homography_complete.py ... --no-internal
```

Disable local refinement of external or internal points:

```bash
python multiaruco_homography_complete.py ... --no-refine-external
python multiaruco_homography_complete.py ... --no-refine-internal
```

## Output

- `*_corrected.png`: corrected frames.
- `homography_validation.csv`: per-frame registration metrics.
- `debug/`: visual point-overlap checks when `--debug-every` is greater than zero.

For the regenerated `SineWaveTest1_corrected_lowerror` output:

- `605/605` frames completed with status `ok`.
- `605` corrected PNG frames were written.
- Mean median inlier error: `0.193786 px`.
- Maximum median inlier error: `0.251911 px`.
- Maximum RMS inlier error: `0.262148 px`.

## Notes

- Use the same ArUco dictionary that was used to print the markers.
- Output frames are saved as PNG to avoid adding JPEG compression artifacts.
- All three markers must lie on the same physical plane for one global
  homography to be valid.
