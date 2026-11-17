"""Stage 6 feature matching with shift alignment between 256x256 palmprint skeletons."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app import config


@dataclass(frozen=True)
class MatchResult:
    """Similarity scores, best translation offset, and verification decision."""

    verified: bool
    score: float
    dilated_iou: float
    dice_f1: float
    hamming_similarity: float
    shift_x: int
    shift_y: int
    reason: str


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


def compute_match_score(
    probe_skeleton: np.ndarray,
    template_skeleton: np.ndarray,
    threshold: float = config.DEFAULT_THRESHOLD,
    dilation_ksize: int = config.DEFAULT_DILATION_KSIZE,
    shift_range: int = config.DEFAULT_SHIFT_RANGE,
    shift_step: int = config.DEFAULT_SHIFT_STEP,
    enable_alignment: bool = True,
) -> MatchResult:
    """Compare two 256x256 binary skeletons with dilation & translation alignment."""
    if probe_skeleton.shape != template_skeleton.shape:
        raise ValueError(
            f"Shape mismatch: probe={probe_skeleton.shape} template={template_skeleton.shape}"
        )

    probe_bin = (probe_skeleton > 0).astype(np.uint8)
    tmpl_bin = (template_skeleton > 0).astype(np.uint8)

    probe_ratio = float(np.count_nonzero(probe_bin)) / float(probe_bin.size)
    tmpl_ratio = float(np.count_nonzero(tmpl_bin)) / float(tmpl_bin.size)

    # Guard against blank/featureless frames accidentally matching or dividing by zero.
    if probe_ratio < config.MIN_SKELETON_RATIO or tmpl_ratio < config.MIN_SKELETON_RATIO:
        return MatchResult(
            verified=False,
            score=0.0,
            dilated_iou=0.0,
            dice_f1=0.0,
            hamming_similarity=0.0,
            shift_x=0,
            shift_y=0,
            reason="insufficient_palm_features",
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

    score = round(dilated_iou, 4)
    verified = score >= threshold
    reason = "matched" if verified else "below_threshold"

    return MatchResult(
        verified=verified,
        score=score,
        dilated_iou=round(dilated_iou, 4),
        dice_f1=round(dice_f1, 4),
        hamming_similarity=round(hamming_sim, 4),
        shift_x=best_dx,
        shift_y=best_dy,
        reason=reason,
    )


def render_alignment_overlay(
    probe_skeleton: np.ndarray,
    template_skeleton: np.ndarray,
    shift_x: int,
    shift_y: int,
    dilation_ksize: int = 3,
) -> np.ndarray:
    """Render RGB/BGR overlay: Template=Magenta, Aligned Probe=Cyan, Overlap=White."""
    probe_bin = (probe_skeleton > 0).astype(np.uint8)
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

    canvas = np.zeros((probe_skeleton.shape[0], probe_skeleton.shape[1], 3), dtype=np.uint8)
    canvas[only_tmpl] = (255, 0, 255)   # BGR Magenta
    canvas[only_probe] = (255, 255, 0)  # BGR Cyan
    canvas[overlap] = (255, 255, 255)   # BGR White
    return canvas
