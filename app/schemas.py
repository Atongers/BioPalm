"""Pydantic schemas for BioPalm Auth OpenAPI specification."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QualityMetrics(BaseModel):
    elapsed_ms: float = Field(..., description="Total CV pipeline processing time (ms).")
    mean_brightness: float = Field(..., description="Mean ROI green-channel brightness (0-255).")
    lighting_status: str = Field(..., description="ok | too_dark | overexposed")
    skeleton_pixels: int = Field(..., description="Foreground pixel count in 1-px skeleton.")
    skeleton_density: float = Field(..., description="Ratio of skeleton foreground pixels.")


class MatchBreakdown(BaseModel):
    score: float = Field(..., description="Primary verification score (Dilated IoU, 0.0-1.0).")
    score_percent: float = Field(..., description="Verification score percentage (0-100).")
    dilated_iou: float
    dice_f1: float
    hamming_similarity: float
    threshold: float
    dilation_ksize: int
    shift_x: int = 0
    shift_y: int = 0
    reason: str


class InspectResponse(BaseModel):
    quality: QualityMetrics
    stages: dict[str, str] = Field(
        default_factory=dict,
        description="Base64 PNG previews for stages 1 through 5.",
    )


class UserSummary(BaseModel):
    user_id: str
    display_name: str
    skeleton_pixels: int
    created_at: str
    updated_at: str


class EnrollResponse(BaseModel):
    enrolled: bool
    user: UserSummary
    quality: QualityMetrics
    stages: dict[str, str] = Field(default_factory=dict)


class VerifyResponse(BaseModel):
    verified: bool
    status: str = Field(..., description="TERVERIFIKASI | DITOLAK")
    matched_user: UserSummary | None = None
    match: MatchBreakdown
    quality: QualityMetrics
    template_preview_b64: str | None = None
    alignment_overlay_b64: str | None = None
    stages: dict[str, str] = Field(default_factory=dict)
    vault_unlocked: bool = False


class VaultStatusResponse(BaseModel):
    session_id: str
    status: str = Field(..., description="locked | unlocked")
    created_at: str
    unlocked_by: str | None = None
    score_percent: float | None = None
    unlocked_at: str | None = None
