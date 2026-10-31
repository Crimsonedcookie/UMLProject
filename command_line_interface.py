import argparse
from pathlib import Path

from batch_pipeline import process_frame_sequence
from pipeline_config import PipelineConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Correct a planar frame sequence using IDs 1,2,3, canonical "
            "internal ArUco corners, sub-pixel refinement, and RANSAC homography."
        )
    )

    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--glob", default="*.jpg")
    parser.add_argument("--reference-index", type=int, default=0)
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--ids", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--border-bits", type=int, default=1)

    parser.add_argument("--min-markers", type=int, default=3)
    parser.add_argument("--ransac-threshold", type=float, default=0.4)
    parser.add_argument("--debug-every", type=int, default=0)

    parser.add_argument(
        "--no-internal",
        action="store_true",
        help="Use only the four external corners of each marker.",
    )
    parser.add_argument(
        "--no-refine-external",
        action="store_true",
        help="Do not locally refine external corners.",
    )
    parser.add_argument(
        "--no-refine-internal",
        action="store_true",
        help="Do not locally refine internal corners.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

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
        debug_every=args.debug_every,
    )

    process_frame_sequence(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        glob_pattern=args.glob,
        reference_index=args.reference_index,
        config=config,
    )