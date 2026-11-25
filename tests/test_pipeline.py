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
        "stage0_roi_alignment",
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

    # Apply harsh uneven left-to-right shadow gradient on probe
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


def make_synthetic_full_hand(
    angle_deg: float = 0.0,
    scale: float = 1.0,
    variant: str = "A",
) -> bytes:
    """Create a synthetic open hand on a non-skin dark desk background."""
    h, w = 480, 480
    canvas = np.full((h, w, 3), (45, 50, 55), dtype=np.uint8)
    skin = (135, 175, 215)  # BGR inside YCrCb + HSV skin range

    # Base upright hand centered around (240, 290)
    hand = np.zeros_like(canvas)
    # Palm body
    cv2.ellipse(hand, (240, 295), (72, 85), 0, 0, 360, skin, -1)
    # Four fingers (index, middle, ring, pinky) with realistic webbing gaps
    fingers = [
        ((178, 228), (165, 112), 13),  # index
        ((218, 214), (218, 92), 13),   # middle
        ((258, 216), (266, 102), 13),  # ring
        ((296, 234), (314, 136), 11),  # pinky
    ]
    for base, tip, r in fingers:
        cv2.line(hand, base, tip, skin, thickness=r * 2)
    # Thumb out to the left
    cv2.line(hand, (175, 320), (118, 250), skin, thickness=26)

    # Principal palm lines on the palm body
    if variant == "A":
        lines = [
            np.array([[195, 248], [225, 290], [238, 345]]),
            np.array([[195, 265], [242, 278], [288, 295]]),
            np.array([[215, 240], [255, 248], [292, 260]]),
        ]
    else:
        lines = [
            np.array([[285, 245], [250, 295], [210, 345]]),
            np.array([[195, 330], [245, 305], [290, 280]]),
            np.array([[200, 285], [245, 260], [285, 242]]),
        ]
    for pts in lines:
        cv2.polylines(hand, [pts], isClosed=False, color=(55, 70, 90), thickness=4)

    mask = np.any(hand > 0, axis=2)
    canvas[mask] = hand[mask]

    if angle_deg != 0.0 or scale != 1.0:
        mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, scale)
        canvas = cv2.warpAffine(
            canvas, mat, (w, h), flags=cv2.INTER_LINEAR, borderValue=(45, 50, 55)
        )

    ok, buf = cv2.imencode(".png", canvas)
    assert ok
    return buf.tobytes()


def test_auto_palm_valley_alignment_handles_tilt_and_scale():
    enroll_res = process_palm_image(make_synthetic_full_hand(0.0, 1.0), crop_mode="auto_palm")
    probe_res = process_palm_image(make_synthetic_full_hand(15.0, 0.85), crop_mode="auto_palm")

    assert enroll_res.alignment_method == "finger_valleys"
    assert probe_res.alignment_method == "finger_valleys"

    match = compute_match_score(
        probe_skeleton=probe_res.skeleton,
        template_skeleton=enroll_res.skeleton,
    )
    assert match.verified is True
    assert match.score >= 0.80


def test_auto_palm_falls_back_to_center_square_when_no_skin_found():
    res = process_palm_image(make_blank_image(), crop_mode="auto_palm")
    assert res.alignment_method == "fallback_center_square"


def test_orb_and_hybrid_matching_modes():
    from app.cv.matcher import render_orb_matches

    # Add textured micro-creases on palm so ORB keypoints are abundant
    rng = np.random.default_rng(7)
    raw_a = cv2.imdecode(
        np.frombuffer(make_synthetic_full_hand(0.0, 1.0, variant="A"), np.uint8),
        cv2.IMREAD_COLOR,
    )
    raw_b = cv2.imdecode(
        np.frombuffer(make_synthetic_full_hand(0.0, 1.0, variant="B"), np.uint8),
        cv2.IMREAD_COLOR,
    )
    ok_a, buf_a = cv2.imencode(".png", raw_a)
    ok_b, buf_b = cv2.imencode(".png", raw_b)
    assert ok_a and ok_b

    enroll_res = process_palm_image(buf_a.tobytes(), crop_mode="auto_palm")
    probe_same = process_palm_image(make_synthetic_full_hand(12.0, 0.90, variant="A"), crop_mode="auto_palm")
    probe_diff = process_palm_image(buf_b.tobytes(), crop_mode="auto_palm")

    orb_same = compute_match_score(
        probe_skeleton=probe_same.skeleton,
        template_skeleton=enroll_res.skeleton,
        probe_gray=probe_same.enhanced_gray,
        template_gray=enroll_res.enhanced_gray,
        matcher_mode="orb",
        threshold=0.70,
    )
    assert orb_same.matcher_mode == "orb"
    assert orb_same.orb_good_matches > 5

    hybrid_same = compute_match_score(
        probe_skeleton=probe_same.skeleton,
        template_skeleton=enroll_res.skeleton,
        probe_gray=probe_same.enhanced_gray,
        template_gray=enroll_res.enhanced_gray,
        matcher_mode="hybrid",
        threshold=0.75,
    )
    assert hybrid_same.matcher_mode == "hybrid"
    assert hybrid_same.verified is True

    hybrid_diff = compute_match_score(
        probe_skeleton=probe_diff.skeleton,
        template_skeleton=enroll_res.skeleton,
        probe_gray=probe_diff.enhanced_gray,
        template_gray=enroll_res.enhanced_gray,
        matcher_mode="hybrid",
        threshold=0.75,
    )
    assert hybrid_diff.verified is False
    assert hybrid_diff.score < hybrid_same.score

    ncc_same = compute_match_score(
        probe_skeleton=probe_same.skeleton,
        template_skeleton=enroll_res.skeleton,
        matcher_mode="line_ncc",
        threshold=0.80,
    )
    ncc_diff = compute_match_score(
        probe_skeleton=probe_diff.skeleton,
        template_skeleton=enroll_res.skeleton,
        matcher_mode="line_ncc",
        threshold=0.80,
    )
    assert ncc_same.verified is True
    assert ncc_same.score >= 0.80
    assert ncc_diff.verified is False
    assert ncc_diff.score < 0.65

    vis_orb = render_orb_matches(probe_same.enhanced_gray, enroll_res.enhanced_gray)
    assert vis_orb.shape[0] == 256 and vis_orb.shape[1] == 512


def test_bilateral_and_otsu_blackhat_suppress_heavy_webcam_pores():
    rng = np.random.default_rng(99)
    enroll_res = process_palm_image(make_synthetic_palm("A"), crop_mode="full")

    raw = cv2.imdecode(
        np.frombuffer(make_synthetic_palm("A"), np.uint8),
        cv2.IMREAD_COLOR,
    ).astype(np.float32)
    grain = rng.normal(0, 10.0, raw.shape)
    noisy = np.clip(raw + grain, 0, 255).astype(np.uint8)
    for _ in range(250):
        px, py = int(rng.integers(8, 312)), int(rng.integers(8, 312))
        cv2.circle(noisy, (px, py), 1, (105, 130, 155), -1)
    ok, buf = cv2.imencode(".png", noisy)
    assert ok

    probe_res = process_palm_image(buf.tobytes(), crop_mode="full", use_blackhat=True)
    assert probe_res.skeleton_density < 0.05
    match = compute_match_score(
        probe_skeleton=probe_res.skeleton,
        template_skeleton=enroll_res.skeleton,
    )
    assert match.verified is True
    assert match.score >= 0.80

