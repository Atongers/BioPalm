"""6-stage palmprint CV pipeline for BioPalm Auth (Materi 1-4, 6-7)."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import time
from typing import Literal

import cv2
import numpy as np

from app import config

CropMode = Literal["auto_palm", "center_square", "full", "box"]


@dataclass(frozen=True)
class RoiBox:
    """Pixel bounding box for manual ROI cropping."""

    x: int
    y: int
    w: int
    h: int


@dataclass
class PipelineResult:
    """Outputs and diagnostics from stages 0 through 5."""

    skeleton: np.ndarray
    enhanced_gray: np.ndarray
    alignment_method: str
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


def segment_skin_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Segment hand skin pixels using YCrCb and HSV fusion (Materi 1 & 7)."""
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    mask_ycrcb = cv2.inRange(
        ycrcb, np.array([0, 133, 77], dtype=np.uint8), np.array([255, 173, 127], dtype=np.uint8)
    )

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask_hsv_low = cv2.inRange(
        hsv, np.array([0, 25, 40], dtype=np.uint8), np.array([25, 255, 255], dtype=np.uint8)
    )
    mask_hsv_high = cv2.inRange(
        hsv, np.array([165, 25, 40], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8)
    )
    mask_hsv = cv2.bitwise_or(mask_hsv_low, mask_hsv_high)

    combined = cv2.bitwise_and(mask_ycrcb, mask_hsv)
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_k)
    return cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_k)


def _select_palm_valleys(
    contour: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Find (V1, V2, contour_centroid) for index-middle and ring-pinky valleys."""
    h, w = image_shape
    area = cv2.contourArea(contour)
    if area < 0.04 * float(h * w):
        return None

    bx, by, bw, bh = cv2.boundingRect(contour)
    if bw < 40 or bh < 40:
        return None

    hull_idx = cv2.convexHull(contour, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 4:
        return None

    try:
        defects = cv2.convexityDefects(contour, hull_idx)
    except cv2.error:
        return None
    if defects is None:
        return None

    moments = cv2.moments(contour)
    if moments["m00"] <= 0:
        return None
    centroid = np.array(
        [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
        dtype=np.float64,
    )

    diag = float(np.hypot(bw, bh))
    candidates: list[tuple[np.ndarray, float]] = []
    for row in defects.reshape(-1, 4):
        s_idx, e_idx, f_idx, depth_q8 = row
        depth = depth_q8 / 256.0
        if depth < 0.07 * diag:
            continue

        pt_s = contour[s_idx][0].astype(np.float64)
        pt_e = contour[e_idx][0].astype(np.float64)
        pt_f = contour[f_idx][0].astype(np.float64)

        # Valley must lie in upper 75% of the hand bounding box (fingers point upward)
        if pt_f[1] > by + 0.75 * bh:
            continue

        v1 = pt_s - pt_f
        v2 = pt_e - pt_f
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-3 or n2 < 1e-3:
            continue

        cos_angle = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_angle)))
        if angle_deg <= 75.0:
            candidates.append((pt_f, angle_deg))

    if len(candidates) < 2:
        return None

    candidates.sort(key=lambda item: item[0][0])  # left to right

    # If 3+ valleys exist (often including thumb-index), pick the pair (V1, V2) that
    # best spans the upper webbing (with a middle valley near its midpoint if present).
    best_pair: tuple[np.ndarray, np.ndarray] | None = None
    best_score = -1e9
    pts = [c[0] for c in candidates]
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            p1, p2 = pts[i], pts[j]
            span = float(np.linalg.norm(p2 - p1))
            if span < 0.16 * bw or span > 0.75 * bw:
                continue

            mid = 0.5 * (p1 + p2)
            # Prefer horizontal/slightly tilted upper webbing over steep thumb slope
            y_diff = abs(p2[1] - p1[1])
            avg_y = 0.5 * (p1[1] + p2[1])
            score = span - 1.2 * y_diff - 0.35 * (avg_y - by)

            # Bonus if a third valley (middle-ring) lies close to the midpoint of V1-V2
            for k in range(i + 1, j):
                dist_mid = float(np.linalg.norm(pts[k] - mid))
                if dist_mid < 0.25 * span:
                    score += 0.45 * span
                    break

            if score > best_score:
                best_score = score
                best_pair = (p1, p2)

    if best_pair is None:
        return None
    return best_pair[0], best_pair[1], centroid


def detect_palm_roi_by_valleys(
    image_bgr: np.ndarray,
) -> tuple[np.ndarray, str, np.ndarray]:
    """Auto-align palm ROI via skin segmentation, finger valleys, and Affine rotation."""
    h, w = image_bgr.shape[:2]
    skin_mask = segment_skin_mask(image_bgr)
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        selection = _select_palm_valleys(largest, (h, w))
    else:
        largest = None
        selection = None

    if selection is None:
        fallback_roi = crop_roi(image_bgr, crop_mode="center_square")
        vis = image_bgr.copy()
        side = max(1, int(min(h, w) * config.DEFAULT_CENTER_CROP_RATIO))
        cx, cy = w // 2, h // 2
        x1, y1 = max(0, cx - side // 2), max(0, cy - side // 2)
        cv2.rectangle(vis, (x1, y1), (x1 + side, y1 + side), (0, 165, 255), 2)
        cv2.putText(
            vis,
            "Fallback: center_square",
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 165, 255),
            2,
        )
        return fallback_roi, "fallback_center_square", vis

    v1, v2, centroid = selection
    if v1[0] > v2[0]:
        v1, v2 = v2, v1

    dx = float(v2[0] - v1[0])
    dy = float(v2[1] - v1[1])
    angle_deg = float(np.degrees(np.arctan2(dy, dx)))
    mid = (float((v1[0] + v2[0]) / 2.0), float((v1[1] + v2[1]) / 2.0))
    valley_dist = float(np.hypot(dx, dy))

    rot_mat = cv2.getRotationMatrix2D(mid, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image_bgr,
        rot_mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Project palm centroid into rotated space to ensure we crop toward the palm side
    rot_centroid_y = (
        rot_mat[1, 0] * centroid[0] + rot_mat[1, 1] * centroid[1] + rot_mat[1, 2]
    )
    palm_below = rot_centroid_y >= mid[1]

    roi_side = int(np.clip(round(valley_dist * 2.1), 64, min(w, h)))
    gap = int(round(valley_dist * 0.22))
    x1 = int(np.clip(round(mid[0] - roi_side / 2.0), 0, max(0, w - roi_side)))
    if palm_below:
        y1 = int(np.clip(round(mid[1] + gap), 0, max(0, h - roi_side)))
    else:
        y1 = int(np.clip(round(mid[1] - gap - roi_side), 0, max(0, h - roi_side)))
    roi_bgr = rotated[y1 : y1 + roi_side, x1 : x1 + roi_side]

    # Build diagnostic overlay on original frame showing valleys + tilted ROI box
    vis = image_bgr.copy()
    if largest is not None:
        cv2.drawContours(vis, [largest], -1, (0, 220, 0), 2)
    p1_i = (int(round(v1[0])), int(round(v1[1])))
    p2_i = (int(round(v2[0])), int(round(v2[1])))
    cv2.line(vis, p1_i, p2_i, (255, 255, 0), 2)
    cv2.circle(vis, p1_i, 6, (0, 0, 255), -1)
    cv2.circle(vis, p2_i, 6, (0, 0, 255), -1)

    inv_mat = cv2.invertAffineTransform(rot_mat)
    box_rot = np.array(
        [
            [x1, y1, 1.0],
            [x1 + roi_side, y1, 1.0],
            [x1 + roi_side, y1 + roi_side, 1.0],
            [x1, y1 + roi_side, 1.0],
        ],
        dtype=np.float64,
    )
    box_orig = (inv_mat @ box_rot.T).T.round().astype(np.int32)
    cv2.polylines(vis, [box_orig], isClosed=True, color=(255, 200, 0), thickness=2)
    cv2.putText(
        vis,
        f"Auto-Palm ({angle_deg:+.1f} deg)",
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 180),
        2,
    )

    return roi_bgr, "finger_valleys", vis


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
    """Stage 3 (Materi 2, 3 & 6): Median -> CLAHE -> optional Black-Hat after CLAHE.

    Returns (clahe_gray, ridge_input):
    - clahe_gray: 256x256 CLAHE-equalized grayscale image (used for ORB matching).
    - ridge_input: smoothed image with dark palm valleys ready for adaptive thresholding.
    """
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
        # Invert black-hat ridges so palm lines remain dark valleys on a flat bright
        # background (~255), stripping slow lighting gradients before thresholding.
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
    crop_mode: CropMode = "auto_palm",
    roi_box: RoiBox | None = None,
    use_blackhat: bool = config.DEFAULT_USE_BLACKHAT,
    clip_limit: float = config.CLAHE_CLIP_LIMIT,
    median_ksize: int = config.MEDIAN_KSIZE,
    block_size: int = config.ADAPTIVE_BLOCK_SIZE,
    c_val: int = config.ADAPTIVE_C,
    min_component_px: int = config.MIN_COMPONENT_PIXELS,
    include_stages: bool = True,
) -> PipelineResult:
    """Run stages 0-5 on an uploaded palm image and collect diagnostics."""
    t0 = time.perf_counter()

    bgr = decode_image(image_bytes)
    vis_alignment: np.ndarray | None = None
    if crop_mode == "auto_palm":
        roi_bgr, alignment_method, vis_alignment = detect_palm_roi_by_valleys(bgr)
    else:
        roi_bgr = crop_roi(bgr, crop_mode=crop_mode, roi_box=roi_box)
        alignment_method = crop_mode
        if include_stages:
            vis_alignment = roi_bgr

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
            "stage0_roi_alignment": encode_png_b64(vis_alignment) if vis_alignment is not None else "",
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
        alignment_method=alignment_method,
        elapsed_ms=elapsed_ms,
        mean_brightness=round(mean_brightness, 2),
        lighting_status=lighting_status,
        skeleton_pixels=skeleton_pixels,
        skeleton_density=skeleton_density,
        stages_b64=stages_b64,
    )
