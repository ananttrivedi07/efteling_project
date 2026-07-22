from __future__ import annotations

import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2


# Folder layout, relative to this script:
#   input/                  <- put source videos here
#   pet/                    <- PET clips are written here
#   not_pet/                <- non-PET clips are written here
#   clipper_progress.json   <- automatic resume data
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
PET_DIR = BASE_DIR / "pet"
NOT_PET_DIR = BASE_DIR / "not_pet"
PROGRESS_FILE = BASE_DIR / "clipper_progress.json"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}
WINDOW_NAME = "Trash video clipper v2.1"
MAX_DISPLAY_WIDTH = 1400
MAX_DISPLAY_HEIGHT = 820
PROGRESS_VERSION = 2
APP_VERSION = "2.1"

# cv2.waitKeyEx() key codes on Windows, with common Linux fallbacks.
LEFT_KEYS = {2424832, 65361, ord("j"), ord("J")}
RIGHT_KEYS = {2555904, 65363, ord("l"), ord("L")}
ENTER_KEYS = {10, 13}
SHIFT_EVENT_KEYS = {16, 160, 161}


def ensure_setup() -> None:
    INPUT_DIR.mkdir(exist_ok=True)
    PET_DIR.mkdir(exist_ok=True)
    NOT_PET_DIR.mkdir(exist_ok=True)

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg was not found in PATH. Install FFmpeg and make sure "
            "the 'ffmpeg' command works in Command Prompt."
        )


def load_progress() -> dict[str, Any]:
    if not PROGRESS_FILE.exists():
        return {"version": PROGRESS_VERSION, "videos": {}}

    try:
        data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        backup = PROGRESS_FILE.with_suffix(".broken.json")
        try:
            shutil.copy2(PROGRESS_FILE, backup)
        except OSError:
            pass
        print(f"WARNING: Could not read progress file: {exc}")
        print(f"Starting a new progress file. A copy was saved as {backup.name}.")
        return {"version": PROGRESS_VERSION, "videos": {}}

    if not isinstance(data, dict) or not isinstance(data.get("videos"), dict):
        return {"version": PROGRESS_VERSION, "videos": {}}

    data["version"] = PROGRESS_VERSION
    return data


def save_progress(progress: dict[str, Any]) -> None:
    temp_path = PROGRESS_FILE.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(progress, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temp_path, PROGRESS_FILE)


def video_fingerprint(video_path: Path) -> dict[str, int]:
    stat = video_path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def format_time(frame_index: int, fps: float) -> str:
    seconds = frame_index / fps
    minutes = int(seconds // 60)
    remaining = seconds - minutes * 60
    return f"{minutes:02d}:{remaining:06.3f}"


def resize_for_display(frame):
    height, width = frame.shape[:2]
    scale = min(
        1.0,
        MAX_DISPLAY_WIDTH / max(width, 1),
        MAX_DISPLAY_HEIGHT / max(height, 1),
    )
    if scale >= 1.0:
        return frame.copy()
    return cv2.resize(
        frame,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def is_shift_held() -> bool:
    """Use Windows' key state so 2x mode lasts exactly while Shift is held."""
    if sys.platform != "win32":
        return False
    return bool(ctypes.windll.user32.GetAsyncKeyState(0x10) & 0x8000)


def draw_overlay(
    frame,
    *,
    video_name: str,
    video_number: int,
    video_count: int,
    current_frame: int,
    total_frames: int,
    segment_start: int,
    fps: float,
    playing: bool,
    fast_play: bool,
    pending_exports: int,
    status: str,
):
    display = resize_for_display(frame)
    height, width = display.shape[:2]

    overlay_height = min(180, height)
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (width, overlay_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, display, 0.32, 0, display)

    if playing:
        state = "PLAYING 2X" if fast_play else "PLAYING 1X"
    else:
        state = "PAUSED"

    current_time = format_time(current_frame, fps)
    total_time = format_time(max(total_frames - 1, 0), fps)
    segment_time = format_time(segment_start, fps)
    export_text = f"background exports: {pending_exports}"

    lines = [
        f"[{video_number}/{video_count}] {video_name}",
        f"{state} | frame {current_frame + 1}/{total_frames} | {current_time} / {total_time}",
        f"next clip starts at frame {segment_start + 1} ({segment_time}) | {export_text}",
        "Space play/pause | hold Shift for 2x | A/D +/-1 sec",
        "arrows or J/L +/-1 frame | P PET | N NOT PET | Enter next | Q quit",
    ]

    y = 24
    for i, line in enumerate(lines):
        scale = 0.60 if i == 0 else 0.52
        thickness = 2 if i == 0 else 1
        cv2.putText(
            display,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += 25

    if status:
        cv2.putText(
            display,
            status[:180],
            (12, min(overlay_height - 8, height - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return display


def read_frame_at(cap: cv2.VideoCapture, frame_index: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def clip_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(0, count)


def discover_old_clips(video_path: Path, total_frames: int) -> list[dict[str, Any]]:
    """Reconstruct resume position from clips made by the previous script."""
    pattern = re.compile(rf"^{re.escape(video_path.stem)}_(\d+)\.mp4$", re.IGNORECASE)
    found: dict[int, tuple[str, Path]] = {}

    for label, folder in (("pet", PET_DIR), ("not_pet", NOT_PET_DIR)):
        for path in folder.glob(f"{video_path.stem}_*.mp4"):
            match = pattern.match(path.name)
            if not match:
                continue
            index = int(match.group(1))
            if index in found:
                print(
                    f"WARNING: Duplicate old clip number {index} for {video_path.name}; "
                    "automatic migration stopped before that clip."
                )
                break
            found[index] = (label, path)

    if not found:
        return []

    clips: list[dict[str, Any]] = []
    next_frame = 0
    expected_index = 1

    for index in sorted(found):
        if index != expected_index:
            print(
                f"WARNING: Missing old clip number {expected_index} for {video_path.name}; "
                "automatic migration stopped at the gap."
            )
            break

        label, path = found[index]
        count = clip_frame_count(path)
        if count <= 0:
            print(f"WARNING: Could not read old clip {path.name}; migration stopped there.")
            break

        end_frame = min(total_frames - 1, next_frame + count - 1)
        clips.append(
            {
                "start_frame": next_frame,
                "end_frame": end_frame,
                "label": label,
                "output": path.relative_to(BASE_DIR).as_posix(),
            }
        )
        next_frame = end_frame + 1
        expected_index += 1
        if next_frame >= total_frames:
            break

    return clips


def get_video_state(
    progress: dict[str, Any],
    video_path: Path,
    total_frames: int,
) -> dict[str, Any]:
    videos = progress["videos"]
    fingerprint = video_fingerprint(video_path)
    existing = videos.get(video_path.name)

    if isinstance(existing, dict) and existing.get("fingerprint") == fingerprint:
        existing.setdefault("clips", [])
        existing.setdefault("next_frame", 0)

        # A prior attempted upgrade may have created an empty progress entry
        # before successfully finding clips made by the original script. Scan
        # the output folders again and repair that empty/stale entry when the
        # files show that more work has already been completed.
        disk_clips = discover_old_clips(video_path, total_frames)
        existing_clips = existing["clips"]
        existing_next = int(existing["next_frame"])
        disk_next = int(disk_clips[-1]["end_frame"]) + 1 if disk_clips else 0

        if disk_clips and disk_next > existing_next:
            existing["clips"] = disk_clips
            existing["next_frame"] = disk_next
            save_progress(progress)
            print(
                f"Recovered {len(disk_clips)} existing clips for {video_path.name}; "
                f"resuming at frame {disk_next + 1}."
            )
            return existing

        clip_count = len(existing_clips)
        print(
            f"Loaded saved progress for {video_path.name}: {clip_count} clip(s); "
            f"resuming at frame {existing_next + 1}."
        )
        return existing

    if existing is not None:
        print(f"NOTICE: {video_path.name} changed since it was labelled; starting it as a new source.")

    migrated_clips = discover_old_clips(video_path, total_frames)
    next_frame = (
        int(migrated_clips[-1]["end_frame"]) + 1 if migrated_clips else 0
    )
    state = {
        "fingerprint": fingerprint,
        "next_frame": next_frame,
        "clips": migrated_clips,
    }
    videos[video_path.name] = state
    save_progress(progress)

    if migrated_clips:
        print(
            f"Recovered {len(migrated_clips)} existing clips for {video_path.name}; "
            f"resuming at frame {next_frame + 1}."
        )
    else:
        print(f"No previous clips found for {video_path.name}; starting at frame 1.")

    return state


def reserved_output_names(progress: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for state in progress.get("videos", {}).values():
        if not isinstance(state, dict):
            continue
        for clip in state.get("clips", []):
            output = clip.get("output")
            if isinstance(output, str):
                names.add(output.lower())
    return names


def next_output_path(
    progress: dict[str, Any],
    label_dir: Path,
    source_stem: str,
) -> Path:
    reserved = reserved_output_names(progress)
    index = 1
    while True:
        candidate = label_dir / f"{source_stem}_{index:04d}.mp4"
        rel = candidate.relative_to(BASE_DIR).as_posix().lower()

        other_dir = NOT_PET_DIR if label_dir == PET_DIR else PET_DIR
        other_candidate = other_dir / candidate.name
        other_rel = other_candidate.relative_to(BASE_DIR).as_posix().lower()

        if (
            not candidate.exists()
            and not other_candidate.exists()
            and rel not in reserved
            and other_rel not in reserved
        ):
            return candidate
        index += 1


def export_clip(
    source: Path,
    output: Path,
    start_frame: int,
    end_frame_inclusive: int,
    fps: float,
) -> None:
    """Export an inclusive frame range using fast accurate input seeking."""
    frame_count = end_frame_inclusive - start_frame + 1
    start_seconds = start_frame / fps
    temp_output = output.with_name(f"{output.stem}.partial{output.suffix}")
    temp_output.unlink(missing_ok=True)

    # With transcoding, FFmpeg's input-side -ss seeks to the requested timestamp
    # and decodes/discards up to it. This is dramatically faster than decoding
    # from frame zero for every clip while retaining frame-accurate starts for
    # normal constant-frame-rate camera videos.
    video_filter = (
        f"setpts=N/{fps:.12g}/TB,"
        "pad=ceil(iw/2)*2:ceil(ih/2)*2"
    )

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_seconds:.12f}",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-vf",
        video_filter,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-r",
        f"{fps:.12g}",
        "-fps_mode",
        "cfr",
        "-frames:v",
        str(frame_count),
        "-movflags",
        "+faststart",
        str(temp_output),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    if result.returncode != 0:
        temp_output.unlink(missing_ok=True)
        error = result.stderr.strip() or "Unknown FFmpeg error"
        raise RuntimeError(error)

    os.replace(temp_output, output)


class ExportManager:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clip-export")
        self._jobs: dict[Path, Future[None]] = {}
        self._messages: list[str] = []
        self._lock = threading.Lock()

    def queue(
        self,
        *,
        source: Path,
        output: Path,
        start_frame: int,
        end_frame: int,
        fps: float,
    ) -> bool:
        output = output.resolve()
        if output.exists():
            return False

        with self._lock:
            current = self._jobs.get(output)
            if current is not None and not current.done():
                return False

            future = self._executor.submit(
                export_clip,
                source,
                output,
                start_frame,
                end_frame,
                fps,
            )
            self._jobs[output] = future
            future.add_done_callback(
                lambda completed, path=output: self._finish_job(path, completed)
            )
        return True

    def _finish_job(self, output: Path, future: Future[None]) -> None:
        try:
            future.result()
            message = f"Finished export: {output.name}"
        except Exception as exc:  # surfaced in the UI; retried next restart
            message = f"EXPORT FAILED ({output.name}): {exc}"

        with self._lock:
            self._messages.append(message)

    def pop_messages(self) -> list[str]:
        with self._lock:
            messages = self._messages[:]
            self._messages.clear()
            return messages

    def pending_count(self) -> int:
        with self._lock:
            return sum(not future.done() for future in self._jobs.values())

    def shutdown(self) -> list[str]:
        self._executor.shutdown(wait=True)
        return self.pop_messages()


def queue_missing_exports(
    manager: ExportManager,
    video_path: Path,
    state: dict[str, Any],
    fps: float,
) -> int:
    queued = 0
    for clip in state.get("clips", []):
        try:
            output = BASE_DIR / clip["output"]
            if manager.queue(
                source=video_path,
                output=output,
                start_frame=int(clip["start_frame"]),
                end_frame=int(clip["end_frame"]),
                fps=fps,
            ):
                queued += 1
        except (KeyError, TypeError, ValueError):
            print(f"WARNING: Ignoring invalid progress entry for {video_path.name}: {clip}")
    return queued


def process_video(
    video_path: Path,
    video_number: int,
    video_count: int,
    progress: dict[str, Any],
    exporter: ExportManager,
) -> bool:
    """Return False when the user wants to quit the whole program."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open: {video_path.name}")
        return True

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total_frames <= 0:
        print(f"Invalid video metadata: {video_path.name}")
        cap.release()
        return True

    state = get_video_state(progress, video_path, total_frames)
    segment_start = max(0, min(total_frames, int(state.get("next_frame", 0))))

    recovered = queue_missing_exports(exporter, video_path, state, fps)
    if recovered:
        print(f"Re-queued {recovered} missing exports for {video_path.name}.")

    if segment_start >= total_frames:
        print(f"Already fully labelled, skipping: {video_path.name}")
        cap.release()
        return True

    current_frame = segment_start
    playing = False
    status = (
        f"Resumed at frame {segment_start + 1}. Hold Shift during playback for 2x."
        if segment_start > 0
        else "Hold Shift during playback for 2x."
    )
    frame = read_frame_at(cap, current_frame)
    if frame is None:
        print(f"Could not read frame {current_frame + 1}: {video_path.name}")
        cap.release()
        return True

    while True:
        export_messages = exporter.pop_messages()
        if export_messages:
            status = export_messages[-1]
            for message in export_messages:
                print(message)

        fast_play = playing and is_shift_held()
        display = draw_overlay(
            frame,
            video_name=video_path.name,
            video_number=video_number,
            video_count=video_count,
            current_frame=current_frame,
            total_frames=total_frames,
            segment_start=segment_start,
            fps=fps,
            playing=playing,
            fast_play=fast_play,
            pending_exports=exporter.pending_count(),
            status=status,
        )
        cv2.imshow(WINDOW_NAME, display)

        if playing:
            speed = 2.0 if fast_play else 1.0
            delay_ms = max(1, round(1000 / (fps * speed)))
        else:
            delay_ms = 30
        key = cv2.waitKeyEx(delay_ms)
        key_low = key & 0xFF if key != -1 else -1

        # Shift is read through GetAsyncKeyState. Treat its window event as if
        # no command key was pressed so playback continues while it is held.
        no_command = key == -1 or key in SHIFT_EVENT_KEYS or key_low in SHIFT_EVENT_KEYS
        if playing and no_command:
            if current_frame >= total_frames - 1:
                playing = False
                status = "End of video. Label the final segment or press Enter."
                continue

            ok, next_frame = cap.read()
            if ok:
                current_frame += 1
                frame = next_frame
            else:
                playing = False
                status = "Could not read the next frame."
            continue

        if key == -1:
            continue

        if key_low in (ord("q"), ord("Q")):
            cap.release()
            return False

        if key in ENTER_KEYS or key_low in ENTER_KEYS:
            cap.release()
            return True

        if key_low == ord(" "):
            playing = not playing
            status = ""
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + 1)
            continue

        target = None
        if key_low in (ord("a"), ord("A")):
            target = current_frame - max(1, round(fps))
        elif key_low in (ord("d"), ord("D")):
            target = current_frame + max(1, round(fps))
        elif key in LEFT_KEYS or key_low in LEFT_KEYS:
            target = current_frame - 1
        elif key in RIGHT_KEYS or key_low in RIGHT_KEYS:
            target = current_frame + 1

        if target is not None:
            playing = False
            target = max(0, min(total_frames - 1, target))
            new_frame = read_frame_at(cap, target)
            if new_frame is not None:
                current_frame = target
                frame = new_frame
                status = ""
            else:
                status = f"Could not seek to frame {target + 1}."
            continue

        if key_low in (ord("p"), ord("P"), ord("n"), ord("N")):
            if current_frame < segment_start:
                playing = False
                status = "Cannot clip before the next segment start. Scrub forward first."
                continue

            label_is_pet = key_low in (ord("p"), ord("P"))
            label_dir = PET_DIR if label_is_pet else NOT_PET_DIR
            label_key = "pet" if label_is_pet else "not_pet"
            label_name = "PET" if label_is_pet else "NOT PET"
            output_path = next_output_path(progress, label_dir, video_path.stem)
            was_playing = playing
            playing = False

            clip_record = {
                "start_frame": segment_start,
                "end_frame": current_frame,
                "label": label_key,
                "output": output_path.relative_to(BASE_DIR).as_posix(),
            }
            state["clips"].append(clip_record)
            segment_start = current_frame + 1
            state["next_frame"] = segment_start

            # Save the decision before starting FFmpeg. If the program or PC
            # stops unexpectedly, the missing file is automatically re-exported
            # next time without asking you to label the segment again.
            try:
                save_progress(progress)
            except OSError as exc:
                state["clips"].pop()
                segment_start = int(clip_record["start_frame"])
                state["next_frame"] = segment_start
                status = f"Could not save progress; clip was not queued: {exc}"
                continue

            exporter.queue(
                source=video_path,
                output=output_path,
                start_frame=int(clip_record["start_frame"]),
                end_frame=int(clip_record["end_frame"]),
                fps=fps,
            )
            status = (
                f"Queued {label_name}: {output_path.name} "
                f"(exports continue in background)"
            )

            if segment_start >= total_frames:
                playing = False
                status += " | Video fully labelled; press Enter."
                continue

            current_frame = segment_start
            new_frame = read_frame_at(cap, current_frame)
            if new_frame is not None:
                frame = new_frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + 1)
                playing = was_playing
            else:
                playing = False
                status = "Label saved, but the next frame could not be read."


def main() -> None:
    print("=" * 72)
    print(f"TRASH VIDEO CLIPPER v{APP_VERSION} - RESUME + BACKGROUND EXPORT BUILD")
    print(f"Script file: {Path(__file__).name}")
    print(f"Progress file: {PROGRESS_FILE}")
    print("Hold Shift while playing for approximately 2x speed.")
    print("=" * 72)

    try:
        ensure_setup()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        input("Press Enter to close...")
        return

    videos = sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not videos:
        print(f"No videos found in: {INPUT_DIR}")
        print("Put videos in the input folder, then run this script again.")
        input("Press Enter to close...")
        return

    progress = load_progress()
    exporter = ExportManager()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        for number, video_path in enumerate(videos, start=1):
            keep_going = process_video(
                video_path,
                number,
                len(videos),
                progress,
                exporter,
            )
            if not keep_going:
                break
    finally:
        cv2.destroyAllWindows()
        pending = exporter.pending_count()
        if pending:
            print(f"Finishing {pending} background export(s) before closing...")
        for message in exporter.shutdown():
            print(message)


if __name__ == "__main__":
    main()
