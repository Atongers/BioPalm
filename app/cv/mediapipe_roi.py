"""MediaPipe Hand Landmarker Stage-0 palm ROI detector & affine aligner."""

from __future__ import annotations

from pathlib import Path
from threading import Lock

import cv2
import numpy as np

from app import config

_DETECTOR_LOCK = Lock()
_DETECTOR = None

# MediaPipe 21-point hand landmark indices used for palm geometry
_WRIST = 0
_THUMB_CMC = 1
_THUMB_MCP = 2
_INDEX_MCP = 5
_MIDDLE_MCP = 9
_RING_MCP = 13
_PINKY_MCP = 17
_PALM_POLY_INDICES = (_WRIST, _THUMB_CMC, _THUMB_MCP, _INDEX_MCP, _MIDDLE_MCP, _RING_MCP, _PINKY_MCP)
_HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def _get_detector(model_path: Path):
    """Lazily initialize singleton MediaPipe HandLandmarker in IMAGE mode."""
    global _DETECTOR
    if _DETECTOR is not None:
        return _DETECTOR

    with _DETECTOR_LOCK:
        if _DETECTOR is not None:
            return _DETECTOR
        if not model_path.exists():
            raise FileNotFoundError(f"MediaPipe model not found at {model_path}")

        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.45,
            min_hand_presence_confidence=0.45,
        )
        _DETECTOR = mp_vision.HandLandmarker.create_from_options(options)
        return _DETECTOR


def detect_palm_roi_mediapipe(
    image_bgr: np.ndarray,
    model_path: Path | None = None,
) -> tuple[np.ndarray, str, np.ndarray, np.ndarray | None] | None:
    """Detect palm via MediaPipe 21 landmarks, level via Affine rotation, and crop central palm.

    Returns (roi_bgr, alignment_method, vis_bgr, roi_palm_mask) or None if no hand detected.
    """
    path = model_path or config.MEDIAPIPE_MODEL_PATH
    try:
        detector = _get_detector(path)
    except (ImportError, FileNotFoundError, RuntimeError):
        return None

    import mediapipe as mp

    h, w = image_bgr.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if not result.hand_landmarks:
        return None

    landmarks = result.hand_landmarks[0]
    pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float64)

    p_wrist = pts[_WRIST]
    p_index = pts[_INDEX_MCP]
    p_pinky = pts[_PINKY_MCP]

    mid_knuckles = 0.5 * (p_index + p_pinky)
    axis = p_pinky - p_index
    to_wrist = p_wrist - mid_knuckles

    # Ensure knuckle axis cross-product with wrist direction stays positive
    # so both left and right hands align upright with wrist at the bottom.
    cross = axis[0] * to_wrist[1] - axis[1] * to_wrist[0]
    if cross < 0:
        axis = -axis

    angle_deg = float(np.degrees(np.arctan2(axis[1], axis[0])))
    knuckle_span = float(np.linalg.norm(axis))
    palm_length = float(np.linalg.norm(to_wrist))
    if knuckle_span < 20.0 or palm_length < 20.0:
        return None

    pivot = (float(mid_knuckles[0]), float(mid_knuckles[1]))
    rot_mat = cv2.getRotationMatrix2D(pivot, angle_deg, 1.0)
    rotated_bgr = cv2.warpAffine(
        image_bgr,
        rot_mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Build anatomical interior palm mask from palm-root landmarks
    palm_poly = pts[list(_PALM_POLY_INDICES)].round().astype(np.int32)
    palm_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(palm_mask, cv2.convexHull(palm_poly), 255)
    rotated_mask = cv2.warpAffine(
        palm_mask,
        rot_mat,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Project pivot & wrist in rotated space (pivot stays fixed; wrist lies at my + palm_length)
    roi_side = int(
        np.clip(round(min(knuckle_span * 1.32, palm_length * 0.85)), 64, min(w, h))
    )
    center_x = pivot[0]
    center_y = pivot[1] + 0.48 * palm_length

    x1 = int(np.clip(round(center_x - roi_side / 2.0), 0, max(0, w - roi_side)))
    y1 = int(np.clip(round(center_y - roi_side / 2.0), 0, max(0, h - roi_side)))

    roi_bgr = rotated_bgr[y1 : y1 + roi_side, x1 : x1 + roi_side]
    roi_mask = rotated_mask[y1 : y1 + roi_side, x1 : x1 + roi_side]

    # Diagnostic visualization: 21 landmarks + connections + aligned palm ROI box
    vis = image_bgr.copy()
    for a, b in _HAND_CONNECTIONS:
        pa = (int(round(pts[a][0])), int(round(pts[a][1])))
        pb = (int(round(pts[b][0])), int(round(pts[b][1])))
        cv2.line(vis, pa, pb, (0, 200, 120), 2)
    for idx_pt, pt in enumerate(pts):
        color = (0, 0, 255) if idx_pt in (_WRIST, _INDEX_MCP, _PINKY_MCP) else (255, 255, 0)
        radius = 5 if idx_pt in (_WRIST, _INDEX_MCP, _PINKY_MCP) else 3
        cv2.circle(vis, (int(round(pt[0])), int(round(pt[1]))), radius, color, -1)

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
        f"MediaPipe ({angle_deg:+.1f} deg)",
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 180),
        2,
    )

    return roi_bgr, "mediapipe_landmarks", vis, roi_mask
