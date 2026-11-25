"""Stage 6 feature matching: Soft Line-Map NCC + Dilated IoU + ORB Keypoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from app import config

MatcherMode = Literal["line_ncc", "dilated_iou", "orb", "hybrid"]


@dataclass(frozen=True)
class MatchResult:
    """Similarity scores, best alignment transform, and verification decision."""

    verified: bool
    score: float
    line_ncc: float
    dilated_iou: float
    dice_f1: float
    hamming_similarity: float
    shift_x: int
    shift_y: int
    reason: str
    matcher_mode: str = "line_ncc"
    best_angle: float = 0.0
    best_scale: float = 1.0
    orb_score: float = 0.0
    orb_good_matches: int = 0


def _shift_bool(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate a 2D boolean mask by (dx, dy) with zero-padding (no wrap-around)."""
    if dx == 0 and dy == 0:
        return mask
    h, w = mask.shape
    out = np.zeros_like(mask)
    src_x1 = max(0, -dx)
    src_x2 = min(w, w - dx)
    dst_x1 = max(0, dx)
    dst_x2 = min(w, w + dx)

    src_y1 = max(0, -dy)
    src_y2 = min(h, h - dy)
    dst_y1 = max(0, dy)
    dst_y2 = min(h, h + dy)

    if src_x1 < src_x2 and src_y1 < src_y2:
        out[dst_y1:dst_y2, dst_x1:dst_x2] = mask[src_y1:src_y2, src_x1:src_x2]
    return out


def _iou_metrics(probe_bool: np.ndarray, tmpl_bool: np.ndarray) -> tuple[float, float, float]:
    intersection = int(np.count_nonzero(probe_bool & tmpl_bool))
    union = int(np.count_nonzero(probe_bool | tmpl_bool))
    total_fg = int(np.count_nonzero(probe_bool) + np.count_nonzero(tmpl_bool))

    dilated_iou = float(intersection / union) if union > 0 else 0.0
    dice_f1 = float((2.0 * intersection) / total_fg) if total_fg > 0 else 0.0
    hamming_sim = float(np.mean(probe_bool == tmpl_bool))
    return dilated_iou, dice_f1, hamming_sim


def _find_best_alignment(
    probe_bool: np.ndarray,
    tmpl_bool: np.ndarray,
    shift_range: int,
    shift_step: int,
) -> tuple[int, int]:
    """Coarse-to-fine translation search maximizing Dilated IoU within +/-shift_range."""
    best_dx, best_dy = 0, 0
    best_iou = -1.0

    step = max(1, shift_step)
    for dy in range(-shift_range, shift_range + 1, step):
        for dx in range(-shift_range, shift_range + 1, step):
            shifted = _shift_bool(probe_bool, dx, dy)
            inter = int(np.count_nonzero(shifted & tmpl_bool))
            if inter == 0:
                continue
            union = int(np.count_nonzero(shifted | tmpl_bool))
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_dx, best_dy = dx, dy

    if step > 1:
        cx, cy = best_dx, best_dy
        for dy in range(max(-shift_range, cy - step + 1), min(shift_range, cy + step - 1) + 1):
            for dx in range(max(-shift_range, cx - step + 1), min(shift_range, cx + step - 1) + 1):
                shifted = _shift_bool(probe_bool, dx, dy)
                inter = int(np.count_nonzero(shifted & tmpl_bool))
                if inter == 0:
                    continue
                union = int(np.count_nonzero(shifted | tmpl_bool))
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_dx, best_dy = dx, dy

    return best_dx, best_dy


def compute_line_ncc(
    probe_skeleton: np.ndarray,
    template_skeleton: np.ndarray,
    enable_alignment: bool = True,
) -> tuple[float, int, int, float, float]:
    """Compute Gaussian-blurred Line-Map NCC across small scale & rotation variations."""
    k = config.LINE_NCC_BLUR_KSIZE
    sigma = config.LINE_NCC_BLUR_SIGMA
    pm = cv2.GaussianBlur((probe_skeleton > 0).astype(np.float32), (k, k), sigma)
    tm = cv2.GaussianBlur((template_skeleton > 0).astype(np.float32), (k, k), sigma)

    m = config.LINE_NCC_MARGIN
    h, w = tm.shape
    tm_crop = tm[m : h - m, m : w - m]
    if float(np.std(tm_crop)) < 1e-5:
        m = 0
        tm_crop = tm
        if float(np.std(tm_crop)) < 1e-5:
            return 0.0, 0, 0, 0.0, 1.0

    if not enable_alignment:
        res = cv2.matchTemplate(pm[m : h - m, m : w - m], tm_crop, cv2.TM_CCOEFF_NORMED)
        score = float(np.clip(res[0, 0], 0.0, 1.0))
        return round(score, 4), 0, 0, 0.0, 1.0

    best_score = -1.0
    best_dx, best_dy = 0, 0
    best_angle, best_scale = 0.0, 1.0
    center = (w / 2.0, h / 2.0)

    for sc in config.LINE_NCC_SCALES:
        for ang in config.LINE_NCC_ANGLES:
            if sc == 1.0 and ang == 0.0:
                pw = pm
            else:
                mat = cv2.getRotationMatrix2D(center, ang, sc)
                pw = cv2.warpAffine(pm, mat, (w, h), flags=cv2.INTER_LINEAR)

            res = cv2.matchTemplate(pw, tm_crop, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > best_score:
                best_score = float(max_val)
                best_dx = int(m - max_loc[0])
                best_dy = int(m - max_loc[1])
                best_angle = float(ang)
                best_scale = float(sc)

    clamped = round(float(np.clip(best_score, 0.0, 1.0)), 4)
    return clamped, best_dx, best_dy, best_angle, best_scale


def compute_orb_matches(
    probe_gray: np.ndarray,
    template_gray: np.ndarray,
) -> tuple[float, int, tuple[cv2.KeyPoint, ...], tuple[cv2.KeyPoint, ...], list[cv2.DMatch]]:
    """Extract ORB keypoints on 256x256 CLAHE grayscale ROIs and match via Lowe ratio + RANSAC."""
    orb = cv2.ORB_create(
        nfeatures=config.ORB_NFEATURES,
        fastThreshold=config.ORB_FAST_THRESHOLD,
    )
    kp1, des1 = orb.detectAndCompute(probe_gray, None)
    kp2, des2 = orb.detectAndCompute(template_gray, None)

    if (
        des1 is None
        or des2 is None
        or len(kp1) < 2
        or len(kp2) < 2
        or len(des1) < 2
        or len(des2) < 2
    ):
        return 0.0, 0, kp1 or (), kp2 or (), []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)

    good: list[cv2.DMatch] = []
    for pair in knn:
        if len(pair) == 2:
            m, n = pair
            if m.distance < config.ORB_RATIO_THRESH * n.distance:
                good.append(m)

    effective = good
    if len(good) >= 8:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 6.0)
        if mask is not None:
            inliers = [m for m, keep in zip(good, mask.ravel().tolist()) if keep]
            effective = inliers

    effective.sort(key=lambda m: m.distance)
    orb_score = min(1.0, len(effective) / config.ORB_TARGET_INLIERS)
    return round(float(orb_score), 4), len(effective), kp1, kp2, effective


def compute_match_score(
    probe_skeleton: np.ndarray,
    template_skeleton: np.ndarray,
    probe_gray: np.ndarray | None = None,
    template_gray: np.ndarray | None = None,
    matcher_mode: MatcherMode = config.DEFAULT_MATCHER_MODE,  # type: ignore[assignment]
    threshold: float = config.DEFAULT_THRESHOLD,
    dilation_ksize: int = config.DEFAULT_DILATION_KSIZE,
    shift_range: int = config.DEFAULT_SHIFT_RANGE,
    shift_step: int = config.DEFAULT_SHIFT_STEP,
    enable_alignment: bool = True,
) -> MatchResult:
    """Compare probe and template using Line-NCC, Dilated IoU, ORB, or Hybrid fusion."""
    if probe_skeleton.shape != template_skeleton.shape:
        raise ValueError(
            f"Shape mismatch: probe={probe_skeleton.shape} template={template_skeleton.shape}"
        )

    probe_bin = (probe_skeleton > 0).astype(np.uint8)
    tmpl_bin = (template_skeleton > 0).astype(np.uint8)

    probe_ratio = float(np.count_nonzero(probe_bin)) / float(probe_bin.size)
    tmpl_ratio = float(np.count_nonzero(tmpl_bin)) / float(tmpl_bin.size)

    if probe_ratio < config.MIN_SKELETON_RATIO or tmpl_ratio < config.MIN_SKELETON_RATIO:
        return MatchResult(
            verified=False,
            score=0.0,
            line_ncc=0.0,
            dilated_iou=0.0,
            dice_f1=0.0,
            hamming_similarity=0.0,
            shift_x=0,
            shift_y=0,
            reason="insufficient_palm_features",
            matcher_mode=matcher_mode,
        )

    line_ncc, ncc_dx, ncc_dy, best_angle, best_scale = compute_line_ncc(
        probe_skeleton=probe_skeleton,
        template_skeleton=template_skeleton,
        enable_alignment=enable_alignment,
    )

    k = max(1, dilation_ksize)
    if k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        probe_dil = cv2.dilate(probe_bin, kernel)
        tmpl_dil = cv2.dilate(tmpl_bin, kernel)
    else:
        probe_dil = probe_bin
        tmpl_dil = tmpl_bin

    probe_bool = probe_dil > 0
    tmpl_bool = tmpl_dil > 0

    best_dx, best_dy = 0, 0
    if enable_alignment and shift_range > 0:
        best_dx, best_dy = _find_best_alignment(
            probe_bool=probe_bool,
            tmpl_bool=tmpl_bool,
            shift_range=shift_range,
            shift_step=shift_step,
        )

    aligned_probe = _shift_bool(probe_bool, best_dx, best_dy)
    dilated_iou, dice_f1, hamming_sim = _iou_metrics(aligned_probe, tmpl_bool)

    orb_score = 0.0
    orb_good = 0
    has_grays = probe_gray is not None and template_gray is not None
    if matcher_mode in ("orb", "hybrid") and has_grays:
        orb_score, orb_good, _, _, _ = compute_orb_matches(
            probe_gray, template_gray  # type: ignore[arg-type]
        )

    effective_mode = matcher_mode
    out_dx, out_dy = ncc_dx, ncc_dy
    out_angle, out_scale = best_angle, best_scale
    if matcher_mode == "line_ncc":
        score = line_ncc
    elif matcher_mode == "orb" and has_grays:
        score = orb_score
    elif matcher_mode == "hybrid" and has_grays:
        score = round(
            config.HYBRID_LINE_NCC_WEIGHT * line_ncc + config.HYBRID_ORB_WEIGHT * orb_score,
            4,
        )
    elif matcher_mode == "hybrid":
        effective_mode = "line_ncc"
        score = line_ncc
    else:
        effective_mode = "dilated_iou"
        score = round(dilated_iou, 4)
        out_dx, out_dy = best_dx, best_dy
        out_angle, out_scale = 0.0, 1.0

    verified = score >= threshold
    reason = "matched" if verified else "below_threshold"

    return MatchResult(
        verified=verified,
        score=score,
        line_ncc=line_ncc,
        dilated_iou=round(dilated_iou, 4),
        dice_f1=round(dice_f1, 4),
        hamming_similarity=round(hamming_sim, 4),
        shift_x=out_dx,
        shift_y=out_dy,
        reason=reason,
        matcher_mode=effective_mode,
        best_angle=out_angle,
        best_scale=out_scale,
        orb_score=orb_score,
        orb_good_matches=orb_good,
    )


def render_alignment_overlay(
    probe_skeleton: np.ndarray,
    template_skeleton: np.ndarray,
    shift_x: int,
    shift_y: int,
    dilation_ksize: int = 3,
    angle: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    """Render BGR overlay: Template=Magenta, Aligned Probe=Cyan, Overlap=White."""
    h, w = probe_skeleton.shape[:2]
    probe_bin = (probe_skeleton > 0).astype(np.uint8)
    if angle != 0.0 or scale != 1.0:
        mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        probe_bin = cv2.warpAffine(probe_bin, mat, (w, h), flags=cv2.INTER_NEAREST)

    tmpl_bin = (template_skeleton > 0).astype(np.uint8)

    vis_k = max(1, min(dilation_ksize, 5))
    if vis_k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (vis_k, vis_k))
        probe_vis = cv2.dilate(probe_bin, kernel) > 0
        tmpl_vis = cv2.dilate(tmpl_bin, kernel) > 0
    else:
        probe_vis = probe_bin > 0
        tmpl_vis = tmpl_bin > 0

    aligned_probe = _shift_bool(probe_vis, shift_x, shift_y)
    overlap = aligned_probe & tmpl_vis
    only_probe = aligned_probe & ~tmpl_vis
    only_tmpl = tmpl_vis & ~aligned_probe

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[only_tmpl] = (255, 0, 255)   # BGR Magenta
    canvas[only_probe] = (255, 255, 0)  # BGR Cyan
    canvas[overlap] = (255, 255, 255)   # BGR White
    return canvas


def render_orb_matches(
    probe_gray: np.ndarray,
    template_gray: np.ndarray,
    max_draw: int = 40,
) -> np.ndarray:
    """Render side-by-side ORB keypoint match visualization for Layar 2 Inspector."""
    _, _, kp1, kp2, matches = compute_orb_matches(probe_gray, template_gray)
    return cv2.drawMatches(
        probe_gray,
        kp1,
        template_gray,
        kp2,
        matches[:max_draw],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
