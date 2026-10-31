from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from collect_labelled_points import collect_labeled_points
from diagnostics import blank_panel_like
from frame_analysis import analyse_frame
from pipeline_config import PipelineConfig
from utility_funcs import (make_detector, natural_sort_key, resolve_dictionary)

WINDOW_NAME = "Live homography verification"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live visual verification for the multi-ArUco homography pipeline."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(r"/Users/nathanramani/PycharmProjects/UML/SineWaveTest1"),
    )
    parser.add_argument("--glob", default="sine_test1_frame_*.jpg")
    parser.add_argument("--reference-index", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--ids", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--border-bits", type=int, default=1)
    parser.add_argument("--min-markers", type=int, default=3)
    parser.add_argument("--ransac-threshold", type=float, default=0.4)
    parser.add_argument("--display-scale", type=float, default=0.5)
    parser.add_argument("--delay-ms", type=int, default=30)
    parser.add_argument("--snapshot-dir", type=Path, default=None)
    parser.add_argument("--no-internal", action="store_true")
    parser.add_argument("--no-refine-external", action="store_true")
    parser.add_argument("--no-refine-internal", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    frame_paths = sorted(args.input_dir.glob(args.glob), key=natural_sort_key)
    if not frame_paths:
        raise FileNotFoundError(f"No frames matched '{args.glob}' in {args.input_dir}")

    if args.reference_index < 0 or args.reference_index >= len(frame_paths):
        raise IndexError(
            f"reference-index={args.reference_index} is outside 0..{len(frame_paths) - 1}"
        )

    dictionary = resolve_dictionary(args.dictionary)
    detector = make_detector(dictionary)
    config = PipelineConfig(
        dictionary_name=args.dictionary,
        marker_ids=tuple(args.ids),
        border_bits=args.border_bits,
        use_internal_corners=not args.no_internal,
        refine_external_corners=not args.no_refine_external,
        refine_internal_corners=not args.no_refine_internal,
        ransac_reproj_threshold_px=args.ransac_threshold,
        min_markers=args.min_markers,
        min_inlier_markers=args.min_markers,
    )

    reference_path = frame_paths[args.reference_index]
    reference_image = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    if reference_image is None:
        raise RuntimeError(f"Could not read reference frame: {reference_path}")

    reference_points, reference_markers = collect_labeled_points(
        image=reference_image,
        detector=detector,
        dictionary=dictionary,
        config=config,
    )
    missing = sorted(set(config.marker_ids) - set(reference_markers))
    if missing:
        raise RuntimeError(f"Reference frame is missing marker IDs: {missing}")

    snapshot_dir = args.snapshot_dir or (args.input_dir / "live_verify_snapshots")
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    paused = False
    index = int(np.clip(args.start_index, 0, len(frame_paths) - 1))
    cached_index = -1
    cached_mosaic: Optional[np.ndarray] = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        if cached_mosaic is None or cached_index != index:
            frame_path = frame_paths[index]
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                cached_mosaic = blank_panel_like(reference_image, "read failed", frame_path.name)
            else:
                cached_mosaic, _corrected, _stats, _mode = analyse_frame(
                    frame=frame,
                    frame_path=frame_path,
                    frame_index=index,
                    total_frames=len(frame_paths),
                    reference_image=reference_image,
                    reference_points=reference_points,
                    detector=detector,
                    dictionary=dictionary,
                    config=config,
                    display_scale=args.display_scale,
                )
            cached_index = index

        cv2.imshow(WINDOW_NAME, cached_mosaic)
        key = cv2.waitKey(0 if paused else max(1, args.delay_ms)) & 0xFF

        if key in (27, ord("q"), ord("Q")):
            break
        if key == ord(" "):
            paused = not paused
            continue
        if key in (ord("s"), ord("S")):
            out_path = snapshot_dir / f"{frame_paths[index].stem}_verify.png"
            cv2.imwrite(str(out_path), cached_mosaic)
            print(f"Saved snapshot: {out_path}")
            continue
        if key in (ord("n"), ord("N"), 83):
            index = min(index + 1, len(frame_paths) - 1)
            continue
        if key in (ord("p"), ord("P"), 81):
            index = max(index - 1, 0)
            continue

        if not paused:
            index = (index + 1) % len(frame_paths)

    cv2.destroyAllWindows()
