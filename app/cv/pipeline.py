"""6-stage palmprint CV pipeline for BioPalm Auth (Materi 1-4, 6-7)."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import time
from typing import Literal

import cv2
import numpy as np

from app import config

CropMode = Literal["center_square", "full", "box"]


@dataclass(frozen=True)
class RoiBox:
    """Pixel bounding box for manual ROI cropping."""

    x: int
    y: int
    w: int
    h: int


@dataclass
class PipelineResult:
    """Outputs and diagnostics from stages 1 through 5."""

    skeleton: np.ndarray
    enhanced_gray: np.ndarray
    elapsed_ms: float
    mean_brightness: float
    lighting_status: str
    skeleton_pixels: int
    skeleton_density: float
    stages_b64: dict[str, str]


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode raw uploaded image bytes into a BGR OpenCV ndarray."""
    if not image_bytes:
        raise ValueError("Uploaded image is empty.")
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0:
        raise ValueError("Unable to decode image bytes. Use PNG or JPEG.")
    return bgr


def encode_png_b64(image: np.ndarray) -> str:
    """Encode an OpenCV grayscale or BGR image to a PNG base64 data URI."""
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Failed to encode pipeline stage image to PNG.")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def crop_roi(
    image: np.ndarray,
    crop_mode: CropMode = "center_square",
    roi_box: RoiBox | None = None,
    center_ratio: float = config.DEFAULT_CENTER_CROP_RATIO,
) -> np.ndarray:
    """Stage 2 helper: extract the palm Region of Interest."""
    h, w = image.shape[:2]
    if crop_mode == "full":
        return image

    if crop_mode == "box":
        if roi_box is None:
            raise ValueError("roi_box is required when crop_mode='box'.")
        x1 = max(0, min(w - 1, roi_box.x))
        y1 = max(0, min(h - 1, roi_box.y))
        x2 = max(x1 + 1, min(w, roi_box.x + roi_box.w))
        y2 = max(y1 + 1, min(h, roi_box.y + roi_box.h))
        return image[y1:y2, x1:x2]

    if crop_mode == "center_square":
        side = max(1, int(min(h, w) * center_ratio))
        cx, cy = w // 2, h // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        return image[y1 : y1 + side, x1 : x1 + side]

    raise ValueError(f"Unsupported crop_mode: {crop_mode}")


def extract_green_channel(image_bgr: np.ndarray) -> np.ndarray:
    """Stage 1 (Materi 1): extract green channel for highest palm-line contrast."""
    if image_bgr.ndim == 2:
        return image_bgr
    return image_bgr[:, :, 1]


def assess_lighting(gray_roi: np.ndarray) -> tuple[float, str]:
    """Compute mean luminance on the ROI and return lighting quality status."""
    mean_val = float(np.mean(gray_roi))
    if mean_val < config.LIGHTING_DARK_MAX:
        return mean_val, "too_dark"
    if mean_val > config.LIGHTING_BRIGHT_MIN:
        return mean_val, "overexposed"
    return mean_val, "ok"


def resize_bilinear(gray_roi: np.ndarray, size: int = config.IMAGE_SIZE) -> np.ndarray:
    """Stage 2 (Materi 4): normalize ROI dimensions using bilinear interpolation."""
    return cv2.resize(gray_roi, (size, size), interpolation=cv2.INTER_LINEAR)


def enhance_and_smooth(
    gray_256: np.ndarray,
    clip_limit: float = config.CLAHE_CLIP_LIMIT,
    median_ksize: int = config.MEDIAN_KSIZE,
    gaussian_ksize: int = config.GAUSSIAN_KSIZE,
    use_blackhat: bool = config.DEFAULT_USE_BLACKHAT,
) -> tuple[np.ndarray, np.ndarray]:
    """Stage 3 (Materi 2, 3 & 6): Median -> CLAHE -> optional Black-Hat after CLAHE."""
    k = median_ksize if median_ksize % 2 == 1 else median_ksize + 1
    k = max(3, k)
    smoothed = cv2.medianBlur(gray_256, k)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=config.CLAHE_TILE_GRID)
    clahe_gray = clahe.apply(smoothed)

    gk = gaussian_ksize if gaussian_ksize % 2 == 1 else gaussian_ksize + 1
    if use_blackhat:
        bh_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (config.BLACKHAT_KSIZE, config.BLACKHAT_KSIZE)
        )
        blackhat = cv2.morphologyEx(clahe_gray, cv2.MORPH_BLACKHAT, bh_kernel)
        if gk >= 3:
            blackhat = cv2.GaussianBlur(blackhat, (gk, gk), 0)
        ridge_input = cv2.bitwise_not(blackhat)
        return clahe_gray, ridge_input

    if gk >= 3:
        return clahe_gray, cv2.GaussianBlur(clahe_gray, (gk, gk), 0)
    return clahe_gray, clahe_gray


def binarize_adaptive(
    enhanced: np.ndarray,
    block_size: int = config.ADAPTIVE_BLOCK_SIZE,
    c_val: int = config.ADAPTIVE_C,
) -> np.ndarray:
    """Stage 4 (Materi 7): Adaptive Gaussian Thresholding (lines=255, skin=0)."""
    blk = block_size if block_size % 2 == 1 else block_size + 1
    blk = max(3, blk)
    return cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blk,
        c_val,
    )


def zhang_suen_thinning(binary_img: np.ndarray, max_iters: int = 100) -> np.ndarray:
    """Stage 5b (Materi 6): Vectorized Zhang-Suen thinning to a 1-px skeleton."""
    padded = np.zeros((binary_img.shape[0] + 2, binary_img.shape[1] + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = (binary_img > 0).astype(np.uint8)

    for _ in range(max_iters):
        changed = False
        for step in (0, 1):
            p2 = padded[0:-2, 1:-1]
            p3 = padded[0:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, 0:-2]
            p8 = padded[1:-1, 0:-2]
            p9 = padded[0:-2, 0:-2]
            center = padded[1:-1, 1:-1]

            neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
                + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
                + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
                + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
                + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
                + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
                + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
            )

            if step == 0:
                cond_prod = ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
            else:
                cond_prod = ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)

            to_delete = (
                (center == 1)
                & (neighbors >= 2)
                & (neighbors <= 6)
                & (transitions == 1)
                & cond_prod
            )
            if np.any(to_delete):
                center[to_delete] = 0
                changed = True

        if not changed:
            break

    return (padded[1:-1, 1:-1] * 255).astype(np.uint8)


def filter_small_components(
    skeleton: np.ndarray,
    min_pixels: int = config.MIN_COMPONENT_PIXELS,
) -> np.ndarray:
    """Remove isolated short skeleton fragments caused by skin pores or sensor noise."""
    if min_pixels <= 1:
        return skeleton
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        skeleton, connectivity=8
    )
    if num_labels <= 1:
        return skeleton

    areas = stats[:, cv2.CC_STAT_AREA]
    keep_mask = areas >= min_pixels
    keep_mask[0] = False  # label 0 is background
    return np.where(keep_mask[labels], np.uint8(255), np.uint8(0))


def clean_and_thin(
    binary_img: np.ndarray,
    open_ksize: int = config.MORPH_OPEN_KSIZE,
    close_ksize: int = config.MORPH_CLOSE_KSIZE,
    min_component_px: int = config.MIN_COMPONENT_PIXELS,
) -> tuple[np.ndarray, np.ndarray]:
    """Stage 5 (Materi 6): Opening -> Closing -> 1-px Thinning -> Component cleanup."""
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))

    opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, open_kernel)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel)
    raw_skeleton = zhang_suen_thinning(cleaned)
    skeleton = filter_small_components(raw_skeleton, min_pixels=min_component_px)
    return cleaned, skeleton


def process_palm_image(
    image_bytes: bytes,
    crop_mode: CropMode = "center_square",
    roi_box: RoiBox | None = None,
    use_blackhat: bool = config.DEFAULT_USE_BLACKHAT,
    clip_limit: float = config.CLAHE_CLIP_LIMIT,
    median_ksize: int = config.MEDIAN_KSIZE,
    block_size: int = config.ADAPTIVE_BLOCK_SIZE,
    c_val: int = config.ADAPTIVE_C,
    min_component_px: int = config.MIN_COMPONENT_PIXELS,
    include_stages: bool = True,
) -> PipelineResult:
    """Run stages 1-5 on an uploaded palm image and collect diagnostics."""
    t0 = time.perf_counter()

    bgr = decode_image(image_bytes)
    roi_bgr = crop_roi(bgr, crop_mode=crop_mode, roi_box=roi_box)

    # Stage 1: Green channel
    green_roi = extract_green_channel(roi_bgr)
    mean_brightness, lighting_status = assess_lighting(green_roi)

    # Stage 2: Resize 256x256 bilinear
    resized_256 = resize_bilinear(green_roi, size=config.IMAGE_SIZE)

    # Stage 3: Median -> CLAHE -> optional Black-Hat after CLAHE
    enhanced_gray, ridge_input = enhance_and_smooth(
        resized_256,
        clip_limit=clip_limit,
        median_ksize=median_ksize,
        use_blackhat=use_blackhat,
    )

    # Stage 4: Adaptive Gaussian Thresholding
    binary = binarize_adaptive(ridge_input, block_size=block_size, c_val=c_val)

    # Stage 5: Morphology Opening, Closing, Thinning, and fragment filtering
    cleaned, skeleton = clean_and_thin(binary, min_component_px=min_component_px)

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    skeleton_pixels = int(np.count_nonzero(skeleton))
    skeleton_density = round(skeleton_pixels / float(skeleton.size), 5)

    stages_b64: dict[str, str] = {}
    if include_stages:
        stages_b64 = {
            "stage1_green_roi": encode_png_b64(green_roi),
            "stage2_resized_256": encode_png_b64(resized_256),
            "stage3_clahe_median": encode_png_b64(ridge_input),
            "stage4_adaptive_binary": encode_png_b64(binary),
            "stage5a_morph_clean": encode_png_b64(cleaned),
            "stage5b_skeleton_1px": encode_png_b64(skeleton),
        }

    return PipelineResult(
        skeleton=skeleton,
        enhanced_gray=enhanced_gray,
        elapsed_ms=elapsed_ms,
        mean_brightness=round(mean_brightness, 2),
        lighting_status=lighting_status,
        skeleton_pixels=skeleton_pixels,
        skeleton_density=skeleton_density,
        stages_b64=stages_b64,
    )
