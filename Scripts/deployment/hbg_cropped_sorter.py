#!/usr/bin/env python3
"""
Real-time bottle/can sorter for Raspberry Pi.

Pipeline:
    startup empty-slide median background
    -> tight slide-only zone
    -> brightness-compensated frame difference
    -> largest moving contour
    -> padded RGB crop
    -> ONNX classifier
    -> multi-frame confirmation
    -> asynchronous servo movement

The servo is disabled unless --enable-motor is supplied.
"""

from __future__ import annotations

import argparse
import queue
import signal
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

try:
    from picamera2 import Picamera2
    from libcamera import controls
except ImportError:
    Picamera2 = None
    controls = None

try:
    import board
    import neopixel
except ImportError:
    board = None
    neopixel = None

try:
    from gpiozero import AngularServo
except ImportError:
    AngularServo = None


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
NUM_PIXELS = 16
LED_ON_COLOR = (0, 0, 0, 255)
running = True


def request_stop(signum=None, frame=None) -> None:
    global running
    running = False


signal.signal(signal.SIGINT, request_stop)
signal.signal(signal.SIGTERM, request_stop)


class AsyncCameraReader(threading.Thread):
    def __init__(self, camera, swap_rb: bool) -> None:
        super().__init__(daemon=True)
        self.camera = camera
        self.swap_rb = swap_rb
        self.frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()

    def run(self) -> None:
        while running and not self.stop_event.is_set():
            frame = self.camera.capture_array("main")[:, :, :3]
            if self.swap_rb:
                # cv2.cvtColor does this channel swap with OpenCV's optimized
                # (often NEON-accelerated) code instead of a slower NumPy
                # fancy-index gather. ascontiguousarray is a zero-cost no-op
                # when frame is already tightly packed, so this never adds a
                # copy beyond what the old fancy-index version already did.
                frame = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_BGR2RGB)

            if self.frames.full():
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    pass

            try:
                self.frames.put_nowait(frame)
            except queue.Full:
                pass

    def read_latest(self) -> np.ndarray | None:
        try:
            return self.frames.get(timeout=2.0)
        except queue.Empty:
            return None

    def close(self) -> None:
        self.stop_event.set()
        self.join(timeout=1.5)


class ServoSorter(threading.Thread):
    def __init__(
        self,
        pin: int,
        left_angle: float,
        right_angle: float,
        return_delay: float,
        settle_time: float,
        min_pulse_ms: float,
        max_pulse_ms: float,
        detach_after_move: bool,
    ) -> None:
        super().__init__(daemon=True)

        if AngularServo is None:
            raise RuntimeError("gpiozero is unavailable")
        if left_angle == right_angle:
            raise ValueError("Left and right angles must differ")

        self.left_angle = float(left_angle)
        self.right_angle = float(right_angle)
        self.return_delay = max(0.0, float(return_delay))
        self.settle_time = max(0.0, float(settle_time))
        self.detach_after_move = detach_after_move
        self.commands: queue.Queue[object] = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.busy_event = threading.Event()

        self.servo = AngularServo(
            pin,
            min_angle=min(self.left_angle, self.right_angle),
            max_angle=max(self.left_angle, self.right_angle),
            initial_angle=None,
            min_pulse_width=min_pulse_ms / 1000.0,
            max_pulse_width=max_pulse_ms / 1000.0,
        )

        self.servo.angle = self.left_angle
        time.sleep(self.settle_time)
        if self.detach_after_move:
            self.servo.detach()

    def trigger(self) -> bool:
        if self.busy_event.is_set() or not self.commands.empty():
            return False
        try:
            self.commands.put_nowait(object())
            return True
        except queue.Full:
            return False

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.commands.get(timeout=0.1)
            except queue.Empty:
                continue

            if self.stop_event.is_set():
                break

            self.busy_event.set()
            try:
                self.servo.angle = self.right_angle
                print(f"MOTOR RIGHT (TARGET) {self.right_angle:g} degrees", flush=True)

                if self.stop_event.wait(self.settle_time):
                    break
                if self.detach_after_move:
                    self.servo.detach()

                if self.stop_event.wait(self.return_delay):
                    break

                self.servo.angle = self.left_angle
                print(f"MOTOR LEFT (NORMAL) {self.left_angle:g} degrees", flush=True)

                if self.stop_event.wait(self.settle_time):
                    break
                if self.detach_after_move:
                    self.servo.detach()
            finally:
                self.busy_event.clear()

    def close(self) -> None:
        self.stop_event.set()
        try:
            self.commands.put_nowait(object())
        except queue.Full:
            pass

        self.join(timeout=max(1.0, self.return_delay + 2 * self.settle_time + 0.5))

        try:
            self.servo.angle = self.left_angle
            time.sleep(self.settle_time)
            if self.detach_after_move:
                self.servo.detach()
            self.servo.close()
        except Exception as error:
            print(f"Servo cleanup warning: {error}", flush=True)


def require_camera() -> None:
    if Picamera2 is None or controls is None:
        raise RuntimeError("Picamera2/libcamera is unavailable in this environment")


def parse_polygon(text: str) -> np.ndarray:
    values = [int(part.strip()) for part in text.split(",")]
    if len(values) < 6 or len(values) % 2:
        raise ValueError("--zone requires at least three x,y points")
    return np.asarray(values, dtype=np.float32).reshape(-1, 2)


def scale_polygon(
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


def make_mask(
    width: int,
    height: int,
    polygon: np.ndarray,
    erode_pixels: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    if erode_pixels > 0:
        size = 2 * erode_pixels + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.erode(mask, kernel, iterations=1)

    return mask


def setup_light(brightness: float):
    if board is None or neopixel is None:
        raise RuntimeError("NeoPixel dependencies are unavailable")

    pixels = neopixel.NeoPixel(
        board.D18,
        NUM_PIXELS,
        brightness=max(0.0, min(1.0, brightness)),
        auto_write=True,
        pixel_order=neopixel.GRBW,
    )
    pixels.fill(LED_ON_COLOR)
    return pixels


def light_off(pixels) -> None:
    if pixels is None:
        return
    try:
        pixels.fill((0, 0, 0, 0))
    except Exception:
        pass


def configure_camera(args: argparse.Namespace):
    require_camera()
    camera = Picamera2()
    config = camera.create_video_configuration(
        main={"size": (args.width, args.height), "format": "RGB888"},
        controls={"FrameRate": float(args.fps)},
        buffer_count=4,
    )
    camera.configure(config)
    camera.start()
    time.sleep(0.7)

    requested_controls = [
        {"AfMode": controls.AfModeEnum.Manual},
        {"LensPosition": float(args.lens_position)},
        {"AeEnable": False},
        {"ExposureTime": int(args.exposure_us)},
        {"AnalogueGain": float(args.gain)},
        {"AwbEnable": True},
    ]
    for control_set in requested_controls:
        try:
            camera.set_controls(control_set)
        except Exception:
            pass

    frame_duration_us = int(round(1_000_000 / max(1, args.fps)))
    try:
        camera.set_controls({"FrameDurationLimits": (frame_duration_us, frame_duration_us)})
    except Exception:
        pass

    time.sleep(1.0)
    return camera


def get_rgb_frame(camera, swap_rb: bool) -> np.ndarray:
    frame = camera.capture_array("main")[:, :, :3]
    if swap_rb:
        return cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_BGR2RGB)
    return frame


def get_detection_size(width: int, height: int, detection_width: int) -> tuple[int, int]:
    detection_height = max(1, int(round(height * detection_width / width)))
    return detection_width, detection_height


def capture_zone_preview(args: argparse.Namespace) -> int:
    pixels = None
    camera = None
    try:
        if not args.no_light:
            pixels = setup_light(args.light_brightness)

        camera = configure_camera(args)
        frame_rgb = get_rgb_frame(camera, not args.no_swap_rb)
        zone_reference = parse_polygon(args.zone)
        zone_full = scale_polygon(
            zone_reference,
            args.zone_reference_width,
            args.zone_reference_height,
            args.width,
            args.height,
        )

        output_path = Path(args.capture_frame).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        raw_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), raw_bgr)

        overlay = raw_bgr.copy()
        cv2.polylines(overlay, [zone_full], True, (0, 255, 0), 3)
        cv2.putText(
            overlay,
            "CLASSIFICATION ZONE",
            tuple(zone_full[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        overlay_path = output_path.with_name(f"{output_path.stem}_zone{output_path.suffix}")
        cv2.imwrite(str(overlay_path), overlay)

        print(f"Raw frame: {output_path}", flush=True)
        print(f"Zone overlay: {overlay_path}", flush=True)
        return 0
    finally:
        if camera is not None:
            try:
                camera.stop()
            except Exception:
                pass
        light_off(pixels)


def capture_background(
    camera,
    args: argparse.Namespace,
    zone_mask: np.ndarray,
    detect_size: tuple[int, int],
) -> np.ndarray:
    print("\nBACKGROUND WARM-UP", flush=True)
    print("Keep the classification zone completely empty.", flush=True)

    for remaining in range(args.background_countdown, 0, -1):
        print(f"Capturing background in {remaining}...", flush=True)
        time.sleep(1.0)

    samples: list[np.ndarray] = []
    for _ in range(args.background_frames):
        frame_rgb = get_rgb_frame(camera, not args.no_swap_rb)
        small = cv2.resize(frame_rgb, detect_size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        samples.append(gray)
        if args.background_sample_delay > 0:
            time.sleep(args.background_sample_delay)

    background = np.median(np.stack(samples), axis=0).astype(np.uint8)

    if args.background_image:
        output_path = Path(args.background_image).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        masked = cv2.bitwise_and(background, background, mask=zone_mask)
        cv2.imwrite(str(output_path), masked)
        print(f"Saved background: {output_path}", flush=True)

    print(f"Background ready from {len(samples)} frames.\n", flush=True)
    return background


def softmax(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    values -= np.max(values)
    exponentials = np.exp(values)
    return exponentials / np.sum(exponentials)


def output_probabilities(output: np.ndarray) -> np.ndarray:
    values = np.asarray(output, dtype=np.float32).reshape(-1)
    if values.size < 2:
        raise RuntimeError(f"Expected two outputs, got {values}")

    first_two = values[:2]
    if (
        np.all(first_two >= 0.0)
        and np.all(first_two <= 1.0)
        and abs(float(np.sum(first_two)) - 1.0) < 0.02
    ):
        return first_two
    return softmax(first_two)


def load_model(model_path: str, threads: int):
    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, threads)
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3

    session = ort.InferenceSession(
        str(Path(model_path).expanduser()),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    model_input = session.get_inputs()[0]
    shape = model_input.shape
    if len(shape) != 4:
        raise RuntimeError(f"Expected a 4D input, got {shape}")

    if shape[1] == 3:
        layout = "NCHW"
        input_height = int(shape[2]) if isinstance(shape[2], int) else 224
        input_width = int(shape[3]) if isinstance(shape[3], int) else 224
    elif shape[3] == 3:
        layout = "NHWC"
        input_height = int(shape[1]) if isinstance(shape[1], int) else 224
        input_width = int(shape[2]) if isinstance(shape[2], int) else 224
    else:
        raise RuntimeError(f"Could not identify RGB layout from {shape}")

    return session, model_input.name, model_input.type, layout, input_width, input_height


def classify_crop(
    session,
    input_name: str,
    input_type: str,
    layout: str,
    input_width: int,
    input_height: int,
    crop_rgb: np.ndarray,
    normalization: str,
) -> tuple[float, float, float]:
    resized = cv2.resize(crop_rgb, (input_width, input_height), interpolation=cv2.INTER_LINEAR)

    if "uint8" in input_type:
        # resized is already uint8 on the normal path; skip the redundant copy.
        tensor = resized if resized.dtype == np.uint8 else resized.astype(np.uint8)
    else:
        # In-place math (*=, -=, /=) reuses this buffer instead of allocating
        # a fresh array at every step.
        tensor = resized.astype(np.float32)
        tensor /= np.float32(255.0)
        if normalization == "imagenet":
            tensor -= IMAGENET_MEAN
            tensor /= IMAGENET_STD
        elif normalization == "half":
            tensor -= 0.5
            tensor /= 0.5
        elif normalization != "none":
            raise ValueError(f"Unknown normalization: {normalization}")

    if layout == "NCHW":
        tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)

    started = time.perf_counter()
    output = session.run(None, {input_name: tensor})[0]
    inference_ms = (time.perf_counter() - started) * 1000.0
    probabilities = output_probabilities(output)
    return float(probabilities[0]), float(probabilities[1]), inference_ms


def find_motion_box(
    frame_rgb: np.ndarray,
    background_gray: np.ndarray,
    background_mean: float,
    zone_mask: np.ndarray,
    detect_size: tuple[int, int],
    args: argparse.Namespace,
):
    small = cv2.resize(frame_rgb, detect_size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # background_gray/zone_mask are fixed for the whole run, so their masked
    # mean is computed once by the caller instead of recomputed every frame.
    brightness_shift = cv2.mean(blurred, mask=zone_mask)[0] - background_mean
    corrected = cv2.convertScaleAbs(blurred, alpha=1.0, beta=-brightness_shift)
    difference = cv2.absdiff(corrected, background_gray)
    _, foreground = cv2.threshold(
        difference,
        args.difference_threshold,
        255,
        cv2.THRESH_BINARY,
    )
    foreground = cv2.bitwise_and(foreground, foreground, mask=zone_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel, iterations=1)
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=args.close_iterations,
    )
    if args.dilate_iterations > 0:
        foreground = cv2.dilate(foreground, kernel, iterations=args.dilate_iterations)

    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    zone_area = float(cv2.countNonZero(zone_mask))
    max_contour_area = zone_area * args.max_contour_area_frac

    # Single pass: compute each contour's area once instead of twice
    # (once to filter, once again for the max() key).
    contour = None
    contour_area = 0.0
    for candidate in contours:
        area = cv2.contourArea(candidate)
        if args.min_contour_area <= area <= max_contour_area and area > contour_area:
            contour = candidate
            contour_area = area

    if contour is None:
        return None, 0.0, 0.0

    x, y, width, height = cv2.boundingRect(contour)

    if width * height > zone_area * args.max_box_area_frac:
        return None, contour_area, 0.0

    local_difference = difference[y : y + height, x : x + width]
    mean_difference = float(np.mean(local_difference))
    if mean_difference < args.min_mean_difference:
        return None, contour_area, mean_difference

    return (x, y, x + width, y + height), contour_area, mean_difference


def map_and_pad_box(
    box: tuple[int, int, int, int],
    source_width: int,
    source_height: int,
    detect_width: int,
    detect_height: int,
    padding_fraction: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = int(round(x1 * source_width / detect_width))
    x2 = int(round(x2 * source_width / detect_width))
    y1 = int(round(y1 * source_height / detect_height))
    y2 = int(round(y2 * source_height / detect_height))

    width = x2 - x1
    height = y2 - y1
    pad_x = int(width * padding_fraction)
    pad_y = int(height * padding_fraction)

    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(source_width, x2 + pad_x),
        min(source_height, y2 + pad_y),
    )


def save_debug_crop(
    crop_rgb: np.ndarray,
    directory: Path,
    frame_index: int,
    label: str,
    target_probability: float,
    other_probability: float,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = (
        f"{timestamp}_frame_{frame_index:07d}_{label}"
        f"_target_{target_probability:.3f}_other_{other_probability:.3f}.jpg"
    )
    cv2.imwrite(str(directory / filename), cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))


def run_detector(args: argparse.Namespace) -> int:
    pixels = None
    camera = None
    camera_reader = None
    servo_sorter = None

    detect_size = get_detection_size(args.width, args.height, args.detection_width)
    zone_reference = parse_polygon(args.zone)
    zone_detect = scale_polygon(
        zone_reference,
        args.zone_reference_width,
        args.zone_reference_height,
        detect_size[0],
        detect_size[1],
    )
    zone_full = scale_polygon(
        zone_reference,
        args.zone_reference_width,
        args.zone_reference_height,
        args.width,
        args.height,
    )
    zone_mask = make_mask(
        detect_size[0], detect_size[1], zone_detect, args.zone_erode
    )

    session, input_name, input_type, layout, input_width, input_height = load_model(
        args.model, args.threads
    )
    print(
        f"Model input: {input_type} {layout} {input_width}x{input_height}",
        flush=True,
    )

    target_streak = 0
    no_motion_frames = 0
    motor_latched = False
    last_trigger_time = -1e9
    frame_index = 0
    event_probabilities: list[tuple[float, float]] = []
    debug_dir = Path(args.save_crops).expanduser().resolve() if args.save_crops else None

    try:
        if not args.no_light:
            pixels = setup_light(args.light_brightness)

        if args.enable_motor:
            servo_sorter = ServoSorter(
                pin=args.servo_pin,
                left_angle=args.left_angle,
                right_angle=args.right_angle,
                return_delay=args.return_delay,
                settle_time=args.servo_settle,
                min_pulse_ms=args.servo_min_pulse_ms,
                max_pulse_ms=args.servo_max_pulse_ms,
                detach_after_move=not args.keep_servo_attached,
            )
            servo_sorter.start()
            print("Motor enabled", flush=True)
        else:
            print(
                "DRY RUN: motor disabled. Add --enable-motor only after inspection.",
                flush=True,
            )

        camera = configure_camera(args)
        background_gray = capture_background(camera, args, zone_mask, detect_size)
        # background_gray/zone_mask are constant for the run, so this masked
        # mean is computed once here instead of every frame in the hot loop.
        background_mean = float(cv2.mean(background_gray, mask=zone_mask)[0])

        camera_reader = AsyncCameraReader(camera, swap_rb=not args.no_swap_rb)
        camera_reader.start()

        print(
            f"Ready: camera={args.width}x{args.height}@{args.fps}, "
            f"detection={detect_size[0]}x{detect_size[1]}, "
            f"normalization={args.normalization}, "
            f"target_threshold={args.target_threshold:.2f}, "
            f"motor_hits={args.motor_hits}",
            flush=True,
        )
        print(f"Zone: {zone_full.tolist()}", flush=True)

        while running:
            frame_started = time.perf_counter()
            frame_rgb = camera_reader.read_latest()
            if frame_rgb is None:
                continue

            if servo_sorter is not None and servo_sorter.busy_event.is_set():
                continue

            box_detect, contour_area, mean_difference = find_motion_box(
                frame_rgb,
                background_gray,
                background_mean,
                zone_mask,
                detect_size,
                args,
            )

            if box_detect is None:
                target_streak = 0
                no_motion_frames += 1

                if motor_latched and no_motion_frames >= args.motor_clear_frames:
                    motor_latched = False
                    print("MOTOR REARMED", flush=True)

                if event_probabilities and no_motion_frames >= args.event_clear_frames:
                    if args.print_mode == "event":
                        mean_other = float(np.mean([p[0] for p in event_probabilities]))
                        mean_target = float(np.mean([p[1] for p in event_probabilities]))
                        event_label = "TARGET" if mean_target >= mean_other else "OTHER"
                        print(
                            f"EVENT {event_label} frames={len(event_probabilities)} "
                            f"mean_target={mean_target:.3f} mean_other={mean_other:.3f}",
                            flush=True,
                        )
                    event_probabilities = []

                frame_index += 1
                continue

            no_motion_frames = 0
            x1, y1, x2, y2 = map_and_pad_box(
                box_detect,
                args.width,
                args.height,
                detect_size[0],
                detect_size[1],
                args.crop_padding_frac,
            )
            crop_rgb = frame_rgb[y1:y2, x1:x2]
            if crop_rgb.size == 0:
                frame_index += 1
                continue

            other_probability, target_probability, inference_ms = classify_crop(
                session,
                input_name,
                input_type,
                layout,
                input_width,
                input_height,
                crop_rgb,
                args.normalization,
            )

            if target_probability >= args.target_threshold:
                reported_label = "TARGET"
                confidence = target_probability
                target_streak += 1
            elif other_probability >= args.other_threshold:
                reported_label = "OTHER"
                confidence = other_probability
                target_streak = 0
            else:
                reported_label = "UNCERTAIN"
                confidence = max(target_probability, other_probability)
                target_streak = 0

            event_probabilities.append((other_probability, target_probability))

            if debug_dir is not None:
                should_save = (
                    args.save_crop_mode == "all"
                    or (args.save_crop_mode == "target" and reported_label == "TARGET")
                    or (args.save_crop_mode == "uncertain" and reported_label == "UNCERTAIN")
                )
                if should_save:
                    save_debug_crop(
                        crop_rgb,
                        debug_dir,
                        frame_index,
                        reported_label,
                        target_probability,
                        other_probability,
                    )

            if reported_label == "TARGET":
                now = time.monotonic()
                cooldown_finished = now - last_trigger_time >= args.motor_cooldown
                if (
                    not motor_latched
                    and target_streak >= args.motor_hits
                    and cooldown_finished
                ):
                    if servo_sorter is None:
                        print(
                            f"TARGET WOULD TRIGGER MOTOR target={target_probability:.3f} "
                            f"frame={frame_index}",
                            flush=True,
                        )
                        motor_latched = True
                        last_trigger_time = now
                    elif servo_sorter.trigger():
                        print(
                            f"MOTOR TRIGGER TARGET target={target_probability:.3f} "
                            f"frame={frame_index}",
                            flush=True,
                        )
                        motor_latched = True
                        last_trigger_time = now

            total_ms = (time.perf_counter() - frame_started) * 1000.0
            if args.print_mode == "frame" and not args.quiet_frames:
                print(
                    f"{reported_label} conf={confidence:.3f} "
                    f"target={target_probability:.3f} other={other_probability:.3f} "
                    f"streak={target_streak} frame={frame_index} "
                    f"infer_ms={inference_ms:.1f} total_ms={total_ms:.1f} "
                    f"area={contour_area:.0f} diff={mean_difference:.1f} "
                    f"box={x1},{y1},{x2},{y2}",
                    flush=True,
                )

            frame_index += 1

        return 0
    finally:
        if camera_reader is not None:
            camera_reader.close()
        if camera is not None:
            try:
                camera.stop()
            except Exception:
                pass
        if servo_sorter is not None:
            servo_sorter.close()
        light_off(pixels)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real-time cropped-input sorter")
    parser.add_argument("--model")
    parser.add_argument(
        "--capture-frame",
        help="Capture a raw frame and zone-overlay image, then exit",
    )

    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--detection-width", type=int, default=320)

    parser.add_argument(
        "--zone",
        default="187,53,447,53,387,364,73,364",
        help="Slide-only classification polygon at the reference resolution",
    )
    parser.add_argument("--zone-reference-width", type=int, default=640)
    parser.add_argument("--zone-reference-height", type=int, default=480)
    parser.add_argument("--zone-erode", type=int, default=1)

    parser.add_argument("--background-countdown", type=int, default=3)
    parser.add_argument("--background-frames", type=int, default=45)
    parser.add_argument("--background-sample-delay", type=float, default=0.015)
    parser.add_argument("--background-image")

    parser.add_argument("--difference-threshold", type=int, default=20)
    parser.add_argument("--min-contour-area", type=float, default=450.0)
    parser.add_argument("--max-contour-area-frac", type=float, default=0.70)
    parser.add_argument("--max-box-area-frac", type=float, default=0.45)
    parser.add_argument("--min-mean-difference", type=float, default=12.0)
    parser.add_argument("--close-iterations", type=int, default=2)
    parser.add_argument("--dilate-iterations", type=int, default=1)
    parser.add_argument("--crop-padding-frac", type=float, default=0.20)

    parser.add_argument(
        "--normalization",
        choices=("imagenet", "half", "none"),
        default="imagenet",
    )
    parser.add_argument("--target-threshold", type=float, default=0.80)
    parser.add_argument("--other-threshold", type=float, default=0.80)
    parser.add_argument("--motor-hits", type=int, default=2)
    parser.add_argument("--motor-clear-frames", type=int, default=4)
    parser.add_argument("--event-clear-frames", type=int, default=4)
    parser.add_argument("--motor-cooldown", type=float, default=2.5)

    parser.add_argument("--print-mode", choices=("frame", "event"), default="frame")
    parser.add_argument("--quiet-frames", action="store_true")
    parser.add_argument("--save-crops")
    parser.add_argument(
        "--save-crop-mode",
        choices=("all", "target", "uncertain"),
        default="all",
    )

    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument(
        "--cv2-threads",
        type=int,
        default=0,
        help=(
            "OpenCV's own worker-thread cap for this process. 0 disables "
            "OpenCV's internal thread pool (usually best here: detection "
            "frames are small, so thread setup/teardown can cost more than it "
            "saves, and it otherwise competes with the threads --threads "
            "already gives ONNX Runtime). Positive caps it at N threads; "
            "negative restores OpenCV's normal auto-detected default."
        ),
    )
    parser.add_argument("--lens-position", type=float, default=7.0)
    parser.add_argument("--exposure-us", type=int, default=500)
    parser.add_argument("--gain", type=float, default=8.0)
    parser.add_argument("--no-swap-rb", action="store_true")

    parser.add_argument("--light-brightness", type=float, default=1.0)
    parser.add_argument("--no-light", action="store_true")

    parser.add_argument(
        "--enable-motor",
        action="store_true",
        help="Physically enable the servo; omit for a dry run",
    )
    parser.add_argument("--servo-pin", type=int, default=12)
    parser.add_argument("--left-angle", type=float, default=0.0)
    parser.add_argument("--right-angle", type=float, default=35.0)
    parser.add_argument("--return-delay", type=float, default=2.0)
    parser.add_argument("--servo-settle", type=float, default=0.5)
    parser.add_argument("--servo-min-pulse-ms", type=float, default=1.0)
    parser.add_argument("--servo-max-pulse-ms", type=float, default=2.2)
    parser.add_argument("--keep-servo-attached", action="store_true")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.capture_frame and not args.model:
        parser.error("--model is required unless --capture-frame is used")
    if args.background_frames < 5:
        parser.error("--background-frames must be at least 5")
    if args.motor_hits < 1:
        parser.error("--motor-hits must be at least 1")
    if args.left_angle == args.right_angle:
        parser.error("--left-angle and --right-angle must differ")
    if args.servo_max_pulse_ms <= args.servo_min_pulse_ms:
        parser.error("--servo-max-pulse-ms must exceed --servo-min-pulse-ms")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    cv2.setNumThreads(args.cv2_threads)
    if args.capture_frame:
        return capture_zone_preview(args)
    return run_detector(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0)
