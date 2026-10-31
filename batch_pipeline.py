import csv
from pathlib import Path

import cv2

from collect_labelled_points import collect_labeled_points
from diagnostics import draw_debug_registration
from perform_homography import compute_homography_with_fallback
from pipeline_config import PipelineConfig
from utility_funcs import (
    natural_sort_key,
    resolve_dictionary,
    make_detector,
)


def process_frame_sequence(
    input_dir: Path,
    output_dir: Path,
    glob_pattern: str,
    reference_index: int,
    config: PipelineConfig,
) -> None:
    frame_paths = sorted(input_dir.glob(glob_pattern), key=natural_sort_key)

    if not frame_paths:
        raise FileNotFoundError(
            f"No frames matched '{glob_pattern}' in {input_dir}"
        )

    if reference_index < 0 or reference_index >= len(frame_paths):
        raise IndexError(
            f"reference_index={reference_index} is outside 0..{len(frame_paths)-1}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug"
    if config.debug_every > 0:
        debug_dir.mkdir(parents=True, exist_ok=True)

    dictionary = resolve_dictionary(config.dictionary_name)
    detector = make_detector(dictionary)

    reference_path = frame_paths[reference_index]
    reference_image = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    if reference_image is None:
        raise RuntimeError(f"Could not read reference frame: {reference_path}")

    reference_points, reference_markers = collect_labeled_points(
        image=reference_image,
        detector=detector,
        dictionary=dictionary,
        config=config,
    )

    missing_reference = sorted(set(config.marker_ids) - set(reference_markers))
    if missing_reference:
        raise RuntimeError(
            f"Reference frame is missing marker IDs: {missing_reference}"
        )

    print(f"Reference frame: {reference_path.name}")
    print(f"Dictionary: {config.dictionary_name}")
    print(f"Detected reference IDs: {sorted(reference_markers)}")

    for marker_id in config.marker_ids:
        count = sum(key[0] == marker_id for key in reference_points)
        internal_count = sum(
            key[0] == marker_id and key[1] == "inner"
            for key in reference_points
        )
        print(
            f"ID {marker_id}: {count} total points "
            f"(4 external + {internal_count} internal)"
        )

    height, width = reference_image.shape[:2]
    csv_path = output_dir / "homography_validation.csv"

    fieldnames = [
        "frame_order",
        "input_file",
        "status",
        "mode",
        "detected_marker_ids",
        "num_points",
        "num_inliers",
        "inlier_ratio",
        "mean_error_px",
        "median_error_px",
        "rms_error_px",
        "p95_error_px",
        "max_error_px",
        "output_file",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for frame_order, frame_path in enumerate(frame_paths):
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                print(f"[SKIP] Could not read {frame_path.name}")
                writer.writerow(
                    {
                        "frame_order": frame_order,
                        "input_file": frame_path.name,
                        "status": "read_failed",
                        "mode": "",
                        "detected_marker_ids": "",
                        "num_points": 0,
                        "num_inliers": 0,
                        "inlier_ratio": "",
                        "mean_error_px": "",
                        "median_error_px": "",
                        "rms_error_px": "",
                        "p95_error_px": "",
                        "max_error_px": "",
                        "output_file": "",
                    }
                )
                continue

            current_points, current_markers = collect_labeled_points(
                image=frame,
                detector=detector,
                dictionary=dictionary,
                config=config,
            )

            detected_ids = sorted(current_markers)
            if len(detected_ids) < config.min_markers:
                print(
                    f"[SKIP] {frame_path.name}: only IDs {detected_ids} detected"
                )
                writer.writerow(
                    {
                        "frame_order": frame_order,
                        "input_file": frame_path.name,
                        "status": "insufficient_markers",
                        "mode": "",
                        "detected_marker_ids": ";".join(map(str, detected_ids)),
                        "num_points": 0,
                        "num_inliers": 0,
                        "inlier_ratio": "",
                        "mean_error_px": "",
                        "median_error_px": "",
                        "rms_error_px": "",
                        "p95_error_px": "",
                        "max_error_px": "",
                        "output_file": "",
                    }
                )
                continue

            stats, mode = compute_homography_with_fallback(
                reference_points=reference_points,
                current_points=current_points,
                config=config,
            )

            if stats is None:
                print(f"[SKIP] {frame_path.name}: homography validation failed")
                writer.writerow(
                    {
                        "frame_order": frame_order,
                        "input_file": frame_path.name,
                        "status": "homography_failed",
                        "mode": mode,
                        "detected_marker_ids": ";".join(map(str, detected_ids)),
                        "num_points": 0,
                        "num_inliers": 0,
                        "inlier_ratio": "",
                        "mean_error_px": "",
                        "median_error_px": "",
                        "rms_error_px": "",
                        "p95_error_px": "",
                        "max_error_px": "",
                        "output_file": "",
                    }
                )
                continue

            corrected = cv2.warpPerspective(
                frame,
                stats["H"],
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

            output_name = f"{frame_path.stem}_corrected.png"
            output_path = output_dir / output_name
            if not cv2.imwrite(str(output_path), corrected):
                raise RuntimeError(f"Could not write {output_path}")

            if config.debug_every > 0 and frame_order % config.debug_every == 0:
                debug = draw_debug_registration(corrected, stats)
                cv2.imwrite(
                    str(debug_dir / f"{frame_path.stem}_debug.png"),
                    debug,
                )

            print(
                f"[{frame_order + 1}/{len(frame_paths)}] {frame_path.name} | "
                f"{mode} | inliers {stats['num_inliers']}/{stats['num_points']} | "
                f"median={stats['median_error_px']:.3f}px | "
                f"RMS={stats['rms_error_px']:.3f}px"
            )

            writer.writerow(
                {
                    "frame_order": frame_order,
                    "input_file": frame_path.name,
                    "status": "ok",
                    "mode": mode,
                    "detected_marker_ids": ";".join(map(str, detected_ids)),
                    "num_points": stats["num_points"],
                    "num_inliers": stats["num_inliers"],
                    "inlier_ratio": f"{stats['inlier_ratio']:.6f}",
                    "mean_error_px": f"{stats['mean_error_px']:.6f}",
                    "median_error_px": f"{stats['median_error_px']:.6f}",
                    "rms_error_px": f"{stats['rms_error_px']:.6f}",
                    "p95_error_px": f"{stats['p95_error_px']:.6f}",
                    "max_error_px": f"{stats['max_error_px']:.6f}",
                    "output_file": output_name,
                }
            )

    print(f"\nCompleted. Corrected PNG frames: {output_dir}")
    print(f"Validation report: {csv_path}")
