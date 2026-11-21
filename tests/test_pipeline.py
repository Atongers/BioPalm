"""Unit tests for the 6-stage OpenCV BioPalm pipeline and matcher."""

from __future__ import annotations

import cv2
import numpy as np

from app.cv.matcher import compute_match_score
from app.cv.pipeline import process_palm_image


def make_synthetic_palm(
    variant: str = "A",
    shift_x: int = 0,
    shift_y: int = 0,
    brightness_offset: int = 0,
) -> bytes:
    """Create a synthetic palm image with skin tone, lighting gradient, and dark lines."""
    h, w = 320, 320
    # Warm skin base (BGR) with gentle diagonal illumination gradient
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grad = ((xx + yy) / (h + w)) * 35.0 - 17.0 + brightness_offset
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[:, :, 0] = 135 + grad  # B
    img[:, :, 1] = 170 + grad  # G (highest contrast against dark lines)
    img[:, :, 2] = 205 + grad  # R
    canvas = np.clip(img, 0, 255).astype(np.uint8)

    # Draw 3 principal lines (life, head, heart)
    if variant == "A":
        pts1 = np.array([[85 + shift_x, 90 + shift_y], [130 + shift_x, 145 + shift_y], [150 + shift_x, 225 + shift_y]])
        pts2 = np.array([[90 + shift_x, 120 + shift_y], [160 + shift_x, 138 + shift_y], [225 + shift_x, 165 + shift_y]])
        pts3 = np.array([[105 + shift_x, 82 + shift_y], [170 + shift_x, 96 + shift_y], [230 + shift_x, 112 + shift_y]])
    else:
        pts1 = np.array([[220 + shift_x, 85 + shift_y], [175 + shift_x, 155 + shift_y], [120 + shift_x, 225 + shift_y]])
        pts2 = np.array([[85 + shift_x, 210 + shift_y], [165 + shift_x, 185 + shift_y], [230 + shift_x, 150 + shift_y]])
        pts3 = np.array([[95 + shift_x, 155 + shift_y], [160 + shift_x, 115 + shift_y], [215 + shift_x, 90 + shift_y]])

    for pts in (pts1, pts2, pts3):
        cv2.polylines(canvas, [pts], isClosed=False, color=(45, 55, 70), thickness=5)

    ok, buf = cv2.imencode(".png", canvas)
    assert ok
    return buf.tobytes()


def make_blank_image() -> bytes:
    canvas = np.full((320, 320, 3), 180, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", canvas)
    assert ok
    return buf.tobytes()


def test_pipeline_outputs_binary_256_skeleton_and_stages():
    img_bytes = make_synthetic_palm("A")
    res = process_palm_image(img_bytes, crop_mode="full", include_stages=True)

    assert res.skeleton.shape == (256, 256)
    assert res.skeleton.dtype == np.uint8
    assert res.enhanced_gray.shape == (256, 256)
    assert res.enhanced_gray.dtype == np.uint8
    assert set(np.unique(res.skeleton).tolist()).issubset({0, 255})
    assert res.skeleton_pixels > 100
    assert res.elapsed_ms >= 0.0
    assert set(res.stages_b64.keys()) == {
        "stage1_green_roi",
        "stage2_resized_256",
        "stage3_clahe_median",
        "stage4_adaptive_binary",
        "stage5a_morph_clean",
        "stage5b_skeleton_1px",
    }


def test_same_palm_matches_above_threshold_with_shift_and_lighting():
    enroll_res = process_palm_image(make_synthetic_palm("A"), crop_mode="full")
    probe_res = process_palm_image(
        make_synthetic_palm("A", shift_x=1, shift_y=1, brightness_offset=-20),
        crop_mode="full",
    )

    match = compute_match_score(
        probe_skeleton=probe_res.skeleton,
        template_skeleton=enroll_res.skeleton,
        threshold=0.80,
        dilation_ksize=5,
    )
    assert match.verified is True
    assert match.score >= 0.80
    assert match.reason == "matched"


def test_different_palm_rejected():
    palm_a = process_palm_image(make_synthetic_palm("A"), crop_mode="full")
    palm_b = process_palm_image(make_synthetic_palm("B"), crop_mode="full")

    match = compute_match_score(
        probe_skeleton=palm_b.skeleton,
        template_skeleton=palm_a.skeleton,
        threshold=0.80,
        dilation_ksize=5,
    )
    assert match.verified is False
    assert match.score < 0.80
    assert match.reason == "below_threshold"


def test_blank_image_rejected_for_insufficient_features():
    palm_a = process_palm_image(make_synthetic_palm("A"), crop_mode="full")
    blank = process_palm_image(make_blank_image(), crop_mode="full")

    match = compute_match_score(
        probe_skeleton=blank.skeleton,
        template_skeleton=palm_a.skeleton,
        threshold=0.80,
        dilation_ksize=5,
    )
    assert match.verified is False
    assert match.score == 0.0
    assert match.reason == "insufficient_palm_features"


def test_box_crop_and_blackhat_mode():
    from app.cv.pipeline import RoiBox

    img_bytes = make_synthetic_palm("A")
    res = process_palm_image(
        img_bytes,
        crop_mode="box",
        roi_box=RoiBox(x=40, y=40, w=240, h=240),
        use_blackhat=True,
    )
    assert res.skeleton.shape == (256, 256)
    assert res.skeleton_pixels > 100


def test_noisy_shifted_palm_aligns_and_rejects_different_palm():
    rng = np.random.default_rng(42)

    # Enroll clean palm A
    enroll_res = process_palm_image(make_synthetic_palm("A"), crop_mode="full")

    # Probe: shifted by (+8, -6) + sensor noise + random pore dots
    raw = cv2.imdecode(
        np.frombuffer(make_synthetic_palm("A", shift_x=8, shift_y=-6), np.uint8),
        cv2.IMREAD_COLOR,
    )
    noise = rng.normal(0, 6.0, raw.shape)
    noisy = np.clip(raw.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    for _ in range(80):
        px, py = int(rng.integers(10, 310)), int(rng.integers(10, 310))
        cv2.circle(noisy, (px, py), 1, (80, 95, 110), -1)
    ok, buf = cv2.imencode(".png", noisy)
    assert ok

    probe_res = process_palm_image(buf.tobytes(), crop_mode="full")
    match = compute_match_score(
        probe_skeleton=probe_res.skeleton,
        template_skeleton=enroll_res.skeleton,
    )
    assert match.verified is True
    assert match.score >= 0.80
    assert abs(match.shift_x) > 0 or abs(match.shift_y) > 0

    # Ensure alignment does not falsely accept a different palm
    palm_b = process_palm_image(make_synthetic_palm("B"), crop_mode="full")
    diff_match = compute_match_score(
        probe_skeleton=palm_b.skeleton,
        template_skeleton=enroll_res.skeleton,
    )
    assert diff_match.verified is False
    assert diff_match.score < 0.80


def test_blackhat_after_clahe_survives_strong_shadow_gradient():
    enroll_res = process_palm_image(make_synthetic_palm("A"), crop_mode="full")

    raw = cv2.imdecode(
        np.frombuffer(make_synthetic_palm("A"), np.uint8),
        cv2.IMREAD_COLOR,
    ).astype(np.float32)
    h, w = raw.shape[:2]
    shadow = np.linspace(-65.0, 45.0, w, dtype=np.float32)[None, :, None]
    shadowed = np.clip(raw + shadow, 15, 250).astype(np.uint8)
    ok, buf = cv2.imencode(".png", shadowed)
    assert ok

    probe_res = process_palm_image(buf.tobytes(), crop_mode="full", use_blackhat=True)
    match = compute_match_score(
        probe_skeleton=probe_res.skeleton,
        template_skeleton=enroll_res.skeleton,
    )
    assert match.verified is True
    assert match.score >= 0.80

