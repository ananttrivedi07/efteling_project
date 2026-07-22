#!/usr/bin/env python3
"""
Prepare a clean bottle/can-vs-other dataset from labelled video clips.

Expected layout:

    videos/
        PET/
            clip_001.mp4
        OTHER/
            clip_101.mp4

Key differences from earlier versions:
- Builds a separate background image for every source video.
- Uses a central classification zone fully inside the slide.
- Detects motion events rather than treating every frame independently.
- Supports several same-label objects in one clip.
- Saves the best one or two frames from each event.
- Splits by source video, never by extracted frame.
- Preserves the source video's aspect ratio during detection.

No GPU or model is required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


@dataclass
class Candidate:
    frame_index: int
    timestamp_s: float
    box_detection: tuple[int, int, int, int]
    area: float
    mean_difference: float
    sharpness: float
    score: float


def iter_videos(root: Path) -> Iterable[tuple[str, Path]]:
    for label in ("PET", "OTHER"):
        folder = root / label
        if not folder.exists():
            continue

        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                yield label, path


def parse_points(text: str) -> np.ndarray:
    values = [int(value.strip()) for value in text.split(",")]
    if len(values) < 6 or len(values) % 2:
        raise ValueError("The zone needs at least three x,y points.")
    return np.asarray(values, dtype=np.float32).reshape(-1, 2)


def scale_reference_polygon(
    polygon: np.ndarray,
    reference_width: int,
    reference_height: int,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    scaled = polygon.copy()
    scaled[:, 0] *= target_width / reference_width
    scaled[:, 1] *= target_height / reference_height
    scaled[:, 0] = np.clip(scaled[:, 0], 0, target_width - 1)
    scaled[:, 1] = np.clip(scaled[:, 1], 0, target_height - 1)
    return scaled.astype(np.int32)


def create_mask(width: int, height: int, polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def assign_video_splits(
    videos: list[tuple[str, Path]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[Path, str]:
    rng = random.Random(seed)
    assignments: dict[Path, str] = {}

    for label in ("PET", "OTHER"):
        paths = [path for current_label, path in videos if current_label == label]
        rng.shuffle(paths)

        total = len(paths)
        train_count = min(total, int(round(total * train_ratio)))
        val_count = min(total - train_count, int(round(total * val_ratio)))

        for index, path in enumerate(paths):
            if index < train_count:
                split = "train"
            elif index < train_count + val_count:
                split = "val"
            else:
                split = "test"
            assignments[path] = split

    return assignments


def video_metadata(video_path: Path) -> tuple[int, int, int, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        return width, height, frames, fps
    finally:
        capture.release()


def detection_dimensions(
    source_width: int,
    source_height: int,
    detection_width: int,
) -> tuple[int, int]:
    detection_height = max(
        1,
        int(round(source_height * detection_width / source_width)),
    )
    return detection_width, detection_height


def build_video_background(
    video_path: Path,
    frame_count: int,
    detection_size: tuple[int, int],
    sample_count: int,
) -> np.ndarray:
    """
    Estimate the static scene separately for each clip.

    The temporal median removes objects that appear for less than half of
    the sampled frames, while retaining that clip's own camera framing,
    exposure, cables, floor, and stationary surrounding trash.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    samples: list[np.ndarray] = []
    indices = np.linspace(
        0,
        max(0, frame_count - 1),
        min(sample_count, max(1, frame_count)),
        dtype=int,
    )

    try:
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok or frame is None:
                continue

            frame = cv2.resize(
                frame,
                detection_size,
                interpolation=cv2.INTER_AREA,
            )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            samples.append(gray)
    finally:
        capture.release()

    if len(samples) < 5:
        raise RuntimeError(
            f"Could not obtain enough background samples from {video_path}"
        )

    return np.median(np.stack(samples), axis=0).astype(np.uint8)


def find_candidates(
    video_path: Path,
    background: np.ndarray,
    zone_mask: np.ndarray,
    detection_size: tuple[int, int],
    fps: float,
    process_every: int,
    difference_threshold: int,
    min_contour_area: float,
    max_contour_area_fraction: float,
    max_box_area_fraction: float,
    min_mean_difference: float,
    severe_blur_threshold: float,
) -> list[Candidate]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []

    zone_area = float(cv2.countNonZero(zone_mask))
    max_contour_area = zone_area * max_contour_area_fraction
    max_box_area = zone_area * max_box_area_fraction

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidates: list[Candidate] = []
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            try:
                small = cv2.resize(
                    frame,
                    detection_size,
                    interpolation=cv2.INTER_AREA,
                )
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)

                brightness_shift = (
                    cv2.mean(blurred, mask=zone_mask)[0]
                    - cv2.mean(background, mask=zone_mask)[0]
                )
                corrected = cv2.convertScaleAbs(
                    blurred,
                    alpha=1.0,
                    beta=-brightness_shift,
                )

                difference = cv2.absdiff(corrected, background)
                _, foreground = cv2.threshold(
                    difference,
                    difference_threshold,
                    255,
                    cv2.THRESH_BINARY,
                )
                foreground = cv2.bitwise_and(foreground, zone_mask)

                foreground = cv2.morphologyEx(
                    foreground,
                    cv2.MORPH_OPEN,
                    kernel,
                    iterations=1,
                )
                foreground = cv2.morphologyEx(
                    foreground,
                    cv2.MORPH_CLOSE,
                    kernel,
                    iterations=2,
                )

                contours, _ = cv2.findContours(
                    foreground,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )

                valid = [
                    contour
                    for contour in contours
                    if min_contour_area
                    <= cv2.contourArea(contour)
                    <= max_contour_area
                ]
                if not valid:
                    continue

                contour = max(valid, key=cv2.contourArea)
                area = float(cv2.contourArea(contour))
                x, y, width, height = cv2.boundingRect(contour)

                box_area = float(width * height)
                if box_area > max_box_area:
                    continue

                local_difference = difference[y:y + height, x:x + width]
                mean_difference = float(np.mean(local_difference))
                if mean_difference < min_mean_difference:
                    continue

                crop_gray = gray[y:y + height, x:x + width]
                sharpness = float(
                    cv2.Laplacian(crop_gray, cv2.CV_64F).var()
                )
                if sharpness < severe_blur_threshold:
                    continue

                score = (
                    area * max(mean_difference, 1.0)
                    + min(sharpness, 300.0)
                )

                candidates.append(
                    Candidate(
                        frame_index=frame_index,
                        timestamp_s=frame_index / fps,
                        box_detection=(x, y, x + width, y + height),
                        area=area,
                        mean_difference=mean_difference,
                        sharpness=sharpness,
                        score=score,
                    )
                )

            finally:
                # Read one frame, then cheaply advance over the frames that
                # will not be analyzed. This is much faster than converting
                # every skipped 1080p frame into a NumPy image.
                frame_index += 1
                for _ in range(process_every - 1):
                    if not capture.grab():
                        break
                    frame_index += 1

    finally:
        capture.release()

    return candidates

def group_into_events(
    candidates: list[Candidate],
    maximum_clear_frames: int,
) -> list[list[Candidate]]:
    if not candidates:
        return []

    events: list[list[Candidate]] = []
    current = [candidates[0]]

    for candidate in candidates[1:]:
        previous = current[-1]
        if candidate.frame_index - previous.frame_index <= maximum_clear_frames:
            current.append(candidate)
        else:
            events.append(current)
            current = [candidate]

    events.append(current)
    return events


def select_from_event(
    event: list[Candidate],
    frames_per_event: int,
    minimum_gap_frames: int,
) -> list[Candidate]:
    selected: list[Candidate] = []

    for candidate in sorted(event, key=lambda item: item.score, reverse=True):
        if all(
            abs(candidate.frame_index - chosen.frame_index)
            >= minimum_gap_frames
            for chosen in selected
        ):
            selected.append(candidate)

        if len(selected) >= frames_per_event:
            break

    return sorted(selected, key=lambda item: item.frame_index)


def extract_original_crop(
    video_path: Path,
    frame_index: int,
    box_detection: tuple[int, int, int, int],
    source_size: tuple[int, int],
    detection_size: tuple[int, int],
    padding_fraction: float,
) -> tuple[np.ndarray | None, tuple[int, int, int, int]]:
    source_width, source_height = source_size
    detection_width, detection_height = detection_size

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None, (0, 0, 0, 0)

    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            return None, (0, 0, 0, 0)
    finally:
        capture.release()

    x1, y1, x2, y2 = box_detection
    x1 = int(round(x1 * source_width / detection_width))
    x2 = int(round(x2 * source_width / detection_width))
    y1 = int(round(y1 * source_height / detection_height))
    y2 = int(round(y2 * source_height / detection_height))

    width = x2 - x1
    height = y2 - y1
    pad_x = int(width * padding_fraction)
    pad_y = int(height * padding_fraction)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(source_width, x2 + pad_x)
    y2 = min(source_height, y2 + pad_y)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, (x1, y1, x2, y2)

    return crop, (x1, y1, x2, y2)


def create_contact_sheet(
    image_paths: list[Path],
    output_path: Path,
    title: str,
    sample_count: int = 64,
    tile_size: int = 160,
) -> None:
    if not image_paths:
        return

    selected = image_paths.copy()
    random.Random(42).shuffle(selected)
    selected = selected[:sample_count]

    columns = 8
    rows = math.ceil(len(selected) / columns)
    header = 45

    sheet = np.full(
        (header + rows * tile_size, columns * tile_size, 3),
        245,
        dtype=np.uint8,
    )

    cv2.putText(
        sheet,
        title,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    for index, path in enumerate(selected):
        image = cv2.imread(str(path))
        if image is None:
            continue

        height, width = image.shape[:2]
        scale = min(tile_size / width, tile_size / height)
        resized = cv2.resize(
            image,
            (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )

        tile = np.full((tile_size, tile_size, 3), 230, dtype=np.uint8)
        x = (tile_size - resized.shape[1]) // 2
        y = (tile_size - resized.shape[0]) // 2
        tile[y:y + resized.shape[0], x:x + resized.shape[1]] = resized

        row = index // columns
        column = index % columns
        sheet[
            header + row * tile_size:header + (row + 1) * tile_size,
            column * tile_size:(column + 1) * tile_size,
        ] = tile

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="prepared_dataset_v3")

    # Coordinates measured from the footage supplied by the user.
    # This zone is deliberately inside the slide, excluding cables, floor,
    # stationary tissues, and the hand-release area near the top.
    parser.add_argument(
        "--zone",
        default="560,120,1340,120,1160,820,220,820",
    )
    parser.add_argument("--zone-reference-width", type=int, default=1920)
    parser.add_argument("--zone-reference-height", type=int, default=1080)
    parser.add_argument("--detection-width", type=int, default=320)

    parser.add_argument("--background-samples", type=int, default=20)
    parser.add_argument("--process-every", type=int, default=3)

    parser.add_argument("--difference-threshold", type=int, default=20)
    parser.add_argument("--min-contour-area", type=float, default=450.0)
    parser.add_argument("--max-contour-area-frac", type=float, default=0.70)
    parser.add_argument("--max-box-area-frac", type=float, default=0.45)
    parser.add_argument("--min-mean-difference", type=float, default=12.0)
    parser.add_argument("--severe-blur-threshold", type=float, default=5.0)

    parser.add_argument("--event-clear-frames", type=int, default=10)
    parser.add_argument("--frames-per-event", type=int, default=2)
    parser.add_argument("--minimum-gap-frames", type=int, default=4)
    parser.add_argument("--max-events-per-video", type=int, default=12)
    parser.add_argument("--crop-padding-frac", type=float, default=0.20)

    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    input_root = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()

    videos = list(iter_videos(input_root))
    if not videos:
        raise RuntimeError(
            f"No videos found below {input_root / 'PET'} or "
            f"{input_root / 'OTHER'}"
        )

    split_assignments = assign_video_splits(
        videos,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    reference_zone = parse_points(args.zone)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / "manifest.csv"
    review_paths: dict[tuple[str, str], list[Path]] = {}
    no_events: list[str] = []
    total_saved = 0

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "output_path",
                "source_video",
                "label",
                "split",
                "event_index",
                "frame_index",
                "timestamp_s",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "motion_area",
                "mean_difference",
                "sharpness",
            ],
        )
        writer.writeheader()

        for number, (label, video_path) in enumerate(videos, start=1):
            try:
                source_width, source_height, frame_count, fps = (
                    video_metadata(video_path)
                )
                detection_size = detection_dimensions(
                    source_width,
                    source_height,
                    args.detection_width,
                )

                zone_polygon = scale_reference_polygon(
                    reference_zone,
                    args.zone_reference_width,
                    args.zone_reference_height,
                    detection_size[0],
                    detection_size[1],
                )
                zone_mask = create_mask(
                    detection_size[0],
                    detection_size[1],
                    zone_polygon,
                )

                background = build_video_background(
                    video_path,
                    frame_count,
                    detection_size,
                    args.background_samples,
                )

                candidates = find_candidates(
                    video_path=video_path,
                    background=background,
                    zone_mask=zone_mask,
                    detection_size=detection_size,
                    fps=fps,
                    process_every=args.process_every,
                    difference_threshold=args.difference_threshold,
                    min_contour_area=args.min_contour_area,
                    max_contour_area_fraction=args.max_contour_area_frac,
                    max_box_area_fraction=args.max_box_area_frac,
                    min_mean_difference=args.min_mean_difference,
                    severe_blur_threshold=args.severe_blur_threshold,
                )

                events = group_into_events(
                    candidates,
                    maximum_clear_frames=args.event_clear_frames,
                )
                events = sorted(
                    events,
                    key=lambda event: max(item.score for item in event),
                    reverse=True,
                )[:args.max_events_per_video]
                events = sorted(events, key=lambda event: event[0].frame_index)

                if not events:
                    no_events.append(str(video_path))
                    print(
                        f"[{number}/{len(videos)}] {label} "
                        f"{video_path.name}: no events"
                    )
                    continue

                split = split_assignments[video_path]
                destination = output_root / split / label
                destination.mkdir(parents=True, exist_ok=True)

                video_saved = 0
                for event_index, event in enumerate(events, start=1):
                    selected = select_from_event(
                        event,
                        frames_per_event=args.frames_per_event,
                        minimum_gap_frames=args.minimum_gap_frames,
                    )

                    for candidate in selected:
                        crop, source_box = extract_original_crop(
                            video_path=video_path,
                            frame_index=candidate.frame_index,
                            box_detection=candidate.box_detection,
                            source_size=(source_width, source_height),
                            detection_size=detection_size,
                            padding_fraction=args.crop_padding_frac,
                        )
                        if crop is None:
                            continue

                        output_name = (
                            f"{label.lower()}_{video_path.stem}"
                            f"_event_{event_index:03d}"
                            f"_frame_{candidate.frame_index:06d}.jpg"
                        )
                        output_path = destination / output_name

                        if not cv2.imwrite(
                            str(output_path),
                            crop,
                            [cv2.IMWRITE_JPEG_QUALITY, 95],
                        ):
                            continue

                        x1, y1, x2, y2 = source_box
                        writer.writerow({
                            "output_path": str(
                                output_path.relative_to(output_root)
                            ),
                            "source_video": str(video_path),
                            "label": label,
                            "split": split,
                            "event_index": event_index,
                            "frame_index": candidate.frame_index,
                            "timestamp_s": f"{candidate.timestamp_s:.3f}",
                            "bbox_x1": x1,
                            "bbox_y1": y1,
                            "bbox_x2": x2,
                            "bbox_y2": y2,
                            "motion_area": f"{candidate.area:.1f}",
                            "mean_difference": (
                                f"{candidate.mean_difference:.2f}"
                            ),
                            "sharpness": f"{candidate.sharpness:.2f}",
                        })

                        review_paths.setdefault((split, label), []).append(
                            output_path
                        )
                        video_saved += 1
                        total_saved += 1

                print(
                    f"[{number}/{len(videos)}] {label} "
                    f"{video_path.name}: {len(events)} events, "
                    f"{video_saved} crops"
                )

            except Exception as error:
                no_events.append(f"{video_path} :: ERROR: {error}")
                print(f"[WARN] {video_path}: {error}")

    (output_root / "videos_with_no_events.txt").write_text(
        "\n".join(no_events),
        encoding="utf-8",
    )

    for split in ("train", "val", "test"):
        for label in ("PET", "OTHER"):
            paths = review_paths.get((split, label), [])
            create_contact_sheet(
                paths,
                output_root / "review_sheets" / f"{split}_{label}.jpg",
                f"{split.upper()} - {label} - {len(paths)} crops",
            )

    print()
    print(f"Finished. Saved {total_saved} crops.")
    print(f"Manifest: {manifest_path}")
    print(f"Review sheets: {output_root / 'review_sheets'}")
    print(f"No-event/error videos: {len(no_events)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
