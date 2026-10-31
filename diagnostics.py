from typing import Optional

import cv2
import numpy as np


def draw_debug_registration(
    corrected_image: np.ndarray,
    stats: dict,
) -> np.ndarray:
    debug = corrected_image.copy()

    for index, (reference, transformed) in enumerate(
        zip(stats["reference"], stats["transformed"])
    ):
        is_inlier = bool(stats["mask"][index])
        colour = (0, 255, 0) if is_inlier else (0, 0, 255)

        ref_xy = tuple(np.round(reference).astype(int))
        cur_xy = tuple(np.round(transformed).astype(int))

        cv2.circle(debug, ref_xy, 3, (255, 0, 0), -1)
        cv2.circle(debug, cur_xy, 2, colour, -1)
        cv2.line(debug, ref_xy, cur_xy, colour, 1)

    text = (
        f"inliers {stats['num_inliers']}/{stats['num_points']} | "
        f"median {stats['median_error_px']:.3f} px | "
        f"RMS {stats['rms_error_px']:.3f} px"
    )
    cv2.putText(
        debug,
        text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return debug

def put_status(
    image: np.ndarray,
    lines: list[str],
    origin: tuple[int, int] = (16, 28),
    font_scale: float = 0.62,
) -> None:
    x, y = origin
    line_height = int(round(28 * font_scale / 0.62))

    for line in lines:
        cv2.putText(
            image,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += line_height


def draw_source_overlay(
    frame: np.ndarray,
    markers: dict[int, np.ndarray],
    points: dict[tuple[int, str, int, int], np.ndarray],
    stats: Optional[dict],
    mode: str,
    frame_index: int,
    total_frames: int,
    frame_name: str,
) -> np.ndarray:
    overlay = frame.copy()

    if markers:
        marker_ids = np.array([[marker_id] for marker_id in sorted(markers)], dtype=np.int32)
        marker_corners = [
            markers[marker_id].reshape(1, 4, 2).astype(np.float32)
            for marker_id in sorted(markers)
        ]
        cv2.aruco.drawDetectedMarkers(overlay, marker_corners, marker_ids)

    for key, point in points.items():
        x, y = np.round(point).astype(int)
        if key[1] == "outer":
            cv2.circle(overlay, (x, y), 5, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.circle(overlay, (x, y), 2, (255, 0, 255), -1, cv2.LINE_AA)

    if stats is not None:
        for is_inlier, point in zip(stats["mask"], stats["current"]):
            x, y = np.round(point).astype(int)
            colour = (0, 220, 0) if bool(is_inlier) else (0, 0, 255)
            cv2.circle(overlay, (x, y), 7, colour, 1, cv2.LINE_AA)

    detected = ",".join(str(marker_id) for marker_id in sorted(markers)) or "none"
    if stats is None:
        metrics = "homography: failed"
    else:
        metrics = (
            f"{mode} | inliers {stats['num_inliers']}/{stats['num_points']} | "
            f"median {stats['median_error_px']:.3f}px | RMS {stats['rms_error_px']:.3f}px"
        )

    put_status(
        overlay,
        [
            f"{frame_index + 1}/{total_frames}  {frame_name}",
            f"IDs: {detected}",
            metrics,
        ],
    )
    return overlay


def blank_panel_like(image: np.ndarray, title: str, detail: str = "") -> np.ndarray:
    panel = np.zeros_like(image)
    put_status(panel, [title, detail] if detail else [title], origin=(24, 44), font_scale=0.8)
    return panel

def resize_panel(image: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return image

    height, width = image.shape[:2]
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def make_mosaic(
    source_overlay: np.ndarray,
    corrected: np.ndarray,
    registration_debug: np.ndarray,
    reference: np.ndarray,
    display_scale: float,
) -> np.ndarray:
    panels = [
        ("source + refined points", source_overlay),
        ("corrected warp", corrected),
        ("registration error", registration_debug),
        ("reference", reference),
    ]

    labelled = []
    for title, image in panels:
        panel = image.copy()
        put_status(panel, [title], origin=(16, panel.shape[0] - 18), font_scale=0.55)
        labelled.append(resize_panel(panel, display_scale))

    top = np.hstack([labelled[0], labelled[1]])
    bottom = np.hstack([labelled[2], labelled[3]])
    return np.vstack([top, bottom])