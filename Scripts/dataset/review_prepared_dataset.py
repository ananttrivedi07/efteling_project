#!/usr/bin/env python3
"""
Review a prepared PET/OTHER dataset quickly.

Controls:
  RIGHT / D / SPACE = keep
  LEFT / A / X      = reject (moved, never deleted)
  Q / ESC           = save progress and quit

Rejected files are moved under:
  <root>/_rejected/<split>/<label>/

Run the same command again to resume.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_done(progress_path: Path) -> set[str]:
    if not progress_path.exists():
        return set()

    done = set()
    with progress_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            path = row.get("relative_path")
            if path:
                done.add(path)
    return done


def append_decision(
    progress_path: Path,
    relative_path: str,
    split: str,
    label: str,
    action: str,
) -> None:
    existed = progress_path.exists()
    with progress_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "split", "label", "action"],
        )
        if not existed:
            writer.writeheader()
        writer.writerow({
            "relative_path": relative_path,
            "split": split,
            "label": label,
            "action": action,
        })


def collect_images(root: Path, splits: list[str]) -> list[Path]:
    images = []
    for split in splits:
        for label in ("PET", "OTHER"):
            folder = root / split / label
            if not folder.exists():
                continue
            images.extend(
                sorted(
                    path
                    for path in folder.rglob("*")
                    if path.is_file() and path.suffix.lower() in EXTENSIONS
                )
            )
    return images


def make_display(
    image: np.ndarray,
    split: str,
    label: str,
    filename: str,
    current: int,
    total: int,
) -> np.ndarray:
    image_h, image_w = image.shape[:2]
    max_w, max_h = 1050, 700
    scale = min(max_w / max(1, image_w), max_h / max(1, image_h), 1.0)
    resized = cv2.resize(
        image,
        (max(1, int(image_w * scale)), max(1, int(image_h * scale))),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.full((840, 1050, 3), 28, dtype=np.uint8)
    x = (1050 - resized.shape[1]) // 2
    y = 95 + (700 - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized

    cv2.putText(
        canvas,
        f"{split.upper()} / {label}    {current}/{total}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        filename[:120],
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "KEEP: Right / D / Space     REJECT: Left / A / X     QUIT: Q",
        (20, 820),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=["train", "val", "test"],
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    progress_path = root / "review_progress.csv"
    done = load_done(progress_path)
    images = [
        path
        for path in collect_images(root, args.splits)
        if str(path.relative_to(root)) not in done
    ]

    if not images:
        print("No unreviewed images remain.")
        return 0

    rejected_root = root / "_rejected"
    kept = 0
    rejected = 0

    cv2.namedWindow("Dataset review", cv2.WINDOW_NORMAL)

    for index, path in enumerate(images, start=1):
        relative = path.relative_to(root)
        split, label = relative.parts[0], relative.parts[1]

        image = cv2.imread(str(path))
        if image is None:
            destination = rejected_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(destination))
            append_decision(
                progress_path, str(relative), split, label, "reject"
            )
            rejected += 1
            continue

        display = make_display(
            image, split, label, path.name, index, len(images)
        )
        cv2.imshow("Dataset review", display)

        while True:
            key = cv2.waitKeyEx(0)

            keep = {
                32, ord("d"), ord("D"), ord("k"), ord("K"),
                2555904, 65363, 63235,
            }
            reject = {
                ord("a"), ord("A"), ord("x"), ord("X"),
                2424832, 65361, 63234,
            }
            quit_keys = {ord("q"), ord("Q"), 27}

            if key in keep:
                append_decision(
                    progress_path, str(relative), split, label, "keep"
                )
                kept += 1
                break

            if key in reject:
                destination = rejected_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    destination.unlink()
                shutil.move(str(path), str(destination))
                append_decision(
                    progress_path, str(relative), split, label, "reject"
                )
                rejected += 1
                break

            if key in quit_keys:
                cv2.destroyAllWindows()
                print(f"Kept {kept}; rejected {rejected}.")
                print("Run the same command again to resume.")
                return 0

    cv2.destroyAllWindows()
    print(f"Finished. Kept {kept}; rejected {rejected}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
