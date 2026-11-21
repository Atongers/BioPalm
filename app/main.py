"""FastAPI application and OpenAPI endpoints for BioPalm Auth."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.cv.matcher import MatchResult, compute_match_score, render_alignment_overlay
from app.cv.pipeline import (
    CropMode,
    PipelineResult,
    RoiBox,
    encode_png_b64,
    process_palm_image,
)
from app.schemas import (
    EnrollResponse,
    InspectResponse,
    MatchBreakdown,
    QualityMetrics,
    UserSummary,
    VaultStatusResponse,
    VerifyResponse,
)
from app.storage.templates import TemplateStore, UserRecord, VaultSessionStore


def _build_roi_box(
    crop_mode: CropMode,
    roi_x: int | None,
    roi_y: int | None,
    roi_w: int | None,
    roi_h: int | None,
) -> RoiBox | None:
    if crop_mode != "box":
        return None
    if None in (roi_x, roi_y, roi_w, roi_h):
        raise HTTPException(
            status_code=400,
            detail="roi_x, roi_y, roi_w, and roi_h are required when crop_mode='box'.",
        )
    if roi_w <= 0 or roi_h <= 0:  # type: ignore[operator]
        raise HTTPException(status_code=400, detail="roi_w and roi_h must be > 0.")
    return RoiBox(x=roi_x, y=roi_y, w=roi_w, h=roi_h)  # type: ignore[arg-type]


def _run_pipeline(
    image_bytes: bytes,
    crop_mode: CropMode,
    roi_box: RoiBox | None,
    use_blackhat: bool,
    clip_limit: float,
    median_ksize: int,
    block_size: int,
    c_val: int,
    min_component_px: int,
    include_stages: bool,
) -> PipelineResult:
    try:
        return process_palm_image(
            image_bytes=image_bytes,
            crop_mode=crop_mode,
            roi_box=roi_box,
            use_blackhat=use_blackhat,
            clip_limit=clip_limit,
            median_ksize=median_ksize,
            block_size=block_size,
            c_val=c_val,
            min_component_px=min_component_px,
            include_stages=include_stages,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _to_quality(res: PipelineResult) -> QualityMetrics:
    return QualityMetrics(
        elapsed_ms=res.elapsed_ms,
        mean_brightness=res.mean_brightness,
        lighting_status=res.lighting_status,
        skeleton_pixels=res.skeleton_pixels,
        skeleton_density=res.skeleton_density,
    )


def _to_user_summary(rec: UserRecord) -> UserSummary:
    return UserSummary(
        user_id=rec.user_id,
        display_name=rec.display_name,
        skeleton_pixels=rec.skeleton_pixels,
        created_at=rec.created_at,
        updated_at=rec.updated_at,
    )


def create_app(data_dir: Path | None = None) -> FastAPI:
    """Create and configure the BioPalm Auth FastAPI application."""
    application = FastAPI(
        title="BioPalm Auth Prototype API",
        description=(
            "Prototipe autentikasi biometrik telapak tangan berbasis OpenCV "
            "(6 tahap PCD: Kanal Hijau, Resize Bilinear, CLAHE+Median, "
            "Adaptive Gaussian, Morfologi+Thinning 1px, Dilated IoU)."
        ),
        version="0.1.0",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    template_store = TemplateStore(data_dir=data_dir)
    vault_store = VaultSessionStore()

    if config.STATIC_DIR.exists():
        application.mount(
            "/static",
            StaticFiles(directory=str(config.STATIC_DIR)),
            name="static",
        )

    @application.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        index_path = config.STATIC_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="Web UI not found.")
        return FileResponse(str(index_path))

    @application.get("/health", tags=["System"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "biopalm-auth"}

    @application.post(
        "/api/pipeline/inspect",
        response_model=InspectResponse,
        tags=["CV Pipeline"],
        summary="Preview 6-stage CV pipeline without storing template",
    )
    async def inspect_pipeline(
        file: Annotated[UploadFile, File(description="Palm image (PNG/JPEG)")],
        crop_mode: Annotated[CropMode, Form()] = "center_square",
        roi_x: Annotated[int | None, Form()] = None,
        roi_y: Annotated[int | None, Form()] = None,
        roi_w: Annotated[int | None, Form()] = None,
        roi_h: Annotated[int | None, Form()] = None,
        use_blackhat: Annotated[bool, Form()] = config.DEFAULT_USE_BLACKHAT,
        clip_limit: Annotated[float, Form()] = config.CLAHE_CLIP_LIMIT,
        median_ksize: Annotated[int, Form()] = config.MEDIAN_KSIZE,
        block_size: Annotated[int, Form()] = config.ADAPTIVE_BLOCK_SIZE,
        c_val: Annotated[int, Form()] = config.ADAPTIVE_C,
        min_component_px: Annotated[int, Form(ge=0, le=200)] = config.MIN_COMPONENT_PIXELS,
    ) -> InspectResponse:
        roi_box = _build_roi_box(crop_mode, roi_x, roi_y, roi_w, roi_h)
        image_bytes = await file.read()
        result = _run_pipeline(
            image_bytes=image_bytes,
            crop_mode=crop_mode,
            roi_box=roi_box,
            use_blackhat=use_blackhat,
            clip_limit=clip_limit,
            median_ksize=median_ksize,
            block_size=block_size,
            c_val=c_val,
            min_component_px=min_component_px,
            include_stages=True,
        )
        return InspectResponse(
            quality=_to_quality(result),
            stages=result.stages_b64,
        )

    @application.post(
        "/api/enroll",
        response_model=EnrollResponse,
        tags=["Biometric Auth"],
        summary="Enroll a user's palmprint 1-px skeleton template",
    )
    async def enroll_user(
        user_id: Annotated[str, Form(description="Alphanumeric ID, e.g. 'fathan'")],
        file: Annotated[UploadFile, File(description="Palm image (PNG/JPEG)")],
        display_name: Annotated[str, Form()] = "",
        crop_mode: Annotated[CropMode, Form()] = "center_square",
        roi_x: Annotated[int | None, Form()] = None,
        roi_y: Annotated[int | None, Form()] = None,
        roi_w: Annotated[int | None, Form()] = None,
        roi_h: Annotated[int | None, Form()] = None,
        use_blackhat: Annotated[bool, Form()] = config.DEFAULT_USE_BLACKHAT,
        clip_limit: Annotated[float, Form()] = config.CLAHE_CLIP_LIMIT,
        median_ksize: Annotated[int, Form()] = config.MEDIAN_KSIZE,
        block_size: Annotated[int, Form()] = config.ADAPTIVE_BLOCK_SIZE,
        c_val: Annotated[int, Form()] = config.ADAPTIVE_C,
        min_component_px: Annotated[int, Form(ge=0, le=200)] = config.MIN_COMPONENT_PIXELS,
        include_stages: Annotated[bool, Form()] = True,
    ) -> EnrollResponse:
        roi_box = _build_roi_box(crop_mode, roi_x, roi_y, roi_w, roi_h)
        image_bytes = await file.read()
        result = _run_pipeline(
            image_bytes=image_bytes,
            crop_mode=crop_mode,
            roi_box=roi_box,
            use_blackhat=use_blackhat,
            clip_limit=clip_limit,
            median_ksize=median_ksize,
            block_size=block_size,
            c_val=c_val,
            min_component_px=min_component_px,
            include_stages=include_stages,
        )

        if result.skeleton_density < config.MIN_SKELETON_RATIO:
            raise HTTPException(
                status_code=422,
                detail="Insufficient palm features detected. Reposition palm or adjust lighting.",
            )

        try:
            record = template_store.save_template(
                user_id=user_id,
                display_name=display_name or user_id,
                skeleton=result.skeleton,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return EnrollResponse(
            enrolled=True,
            user=_to_user_summary(record),
            quality=_to_quality(result),
            stages=result.stages_b64,
        )

    @application.post(
        "/api/verify",
        response_model=VerifyResponse,
        tags=["Biometric Auth"],
        summary="Verify a palm image against stored template(s) (1:1 or 1:N)",
    )
    async def verify_palm(
        file: Annotated[UploadFile, File(description="Palm image (PNG/JPEG)")],
        user_id: Annotated[
            str | None,
            Form(description="Target user_id (1:1). Leave empty for 1:N identification."),
        ] = None,
        session_id: Annotated[
            str | None,
            Form(description="Optional Vault session_id to unlock on success."),
        ] = None,
        threshold: Annotated[float, Form(ge=0.0, le=1.0)] = config.DEFAULT_THRESHOLD,
        dilation_ksize: Annotated[int, Form(ge=1, le=21)] = config.DEFAULT_DILATION_KSIZE,
        shift_range: Annotated[int, Form(ge=0, le=30)] = config.DEFAULT_SHIFT_RANGE,
        enable_alignment: Annotated[bool, Form()] = True,
        crop_mode: Annotated[CropMode, Form()] = "center_square",
        roi_x: Annotated[int | None, Form()] = None,
        roi_y: Annotated[int | None, Form()] = None,
        roi_w: Annotated[int | None, Form()] = None,
        roi_h: Annotated[int | None, Form()] = None,
        use_blackhat: Annotated[bool, Form()] = config.DEFAULT_USE_BLACKHAT,
        clip_limit: Annotated[float, Form()] = config.CLAHE_CLIP_LIMIT,
        median_ksize: Annotated[int, Form()] = config.MEDIAN_KSIZE,
        block_size: Annotated[int, Form()] = config.ADAPTIVE_BLOCK_SIZE,
        c_val: Annotated[int, Form()] = config.ADAPTIVE_C,
        min_component_px: Annotated[int, Form(ge=0, le=200)] = config.MIN_COMPONENT_PIXELS,
        include_stages: Annotated[bool, Form()] = True,
    ) -> VerifyResponse:
        candidates: list[tuple[UserRecord, object]] = []
        if user_id and user_id.strip():
            try:
                loaded = template_store.get_template(user_id.strip())
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if loaded is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Enrolled user '{user_id}' not found.",
                )
            candidates.append(loaded)
        else:
            for rec in template_store.list_users():
                loaded = template_store.get_template(rec.user_id)
                if loaded is not None:
                    candidates.append(loaded)
            if not candidates:
                raise HTTPException(
                    status_code=404,
                    detail="No users enrolled yet. Call POST /api/enroll first.",
                )

        roi_box = _build_roi_box(crop_mode, roi_x, roi_y, roi_w, roi_h)
        image_bytes = await file.read()
        result = _run_pipeline(
            image_bytes=image_bytes,
            crop_mode=crop_mode,
            roi_box=roi_box,
            use_blackhat=use_blackhat,
            clip_limit=clip_limit,
            median_ksize=median_ksize,
            block_size=block_size,
            c_val=c_val,
            min_component_px=min_component_px,
            include_stages=include_stages,
        )

        best_record: UserRecord | None = None
        best_template = None
        best_match: MatchResult | None = None

        for rec, tmpl_skeleton in candidates:
            m = compute_match_score(
                probe_skeleton=result.skeleton,
                template_skeleton=tmpl_skeleton,  # type: ignore[arg-type]
                threshold=threshold,
                dilation_ksize=dilation_ksize,
                shift_range=shift_range,
                enable_alignment=enable_alignment,
            )
            if best_match is None or m.score > best_match.score:
                best_match = m
                best_record = rec
                best_template = tmpl_skeleton

        assert best_match is not None and best_record is not None

        vault_unlocked = False
        if best_match.verified and session_id and session_id.strip():
            unlocked = vault_store.unlock_session(
                session_id=session_id.strip(),
                user_id=best_record.user_id,
                score=best_match.score,
            )
            vault_unlocked = unlocked is not None

        template_preview = (
            encode_png_b64(best_template)  # type: ignore[arg-type]
            if include_stages and best_template is not None
            else None
        )
        overlay_b64 = None
        if include_stages and best_template is not None:
            overlay_img = render_alignment_overlay(
                probe_skeleton=result.skeleton,
                template_skeleton=best_template,  # type: ignore[arg-type]
                shift_x=best_match.shift_x,
                shift_y=best_match.shift_y,
                dilation_ksize=dilation_ksize,
            )
            overlay_b64 = encode_png_b64(overlay_img)

        return VerifyResponse(
            verified=best_match.verified,
            status="TERVERIFIKASI" if best_match.verified else "DITOLAK",
            matched_user=_to_user_summary(best_record) if best_match.verified else None,
            match=MatchBreakdown(
                score=best_match.score,
                score_percent=round(best_match.score * 100.0, 2),
                dilated_iou=best_match.dilated_iou,
                dice_f1=best_match.dice_f1,
                hamming_similarity=best_match.hamming_similarity,
                threshold=threshold,
                dilation_ksize=dilation_ksize,
                shift_x=best_match.shift_x,
                shift_y=best_match.shift_y,
                reason=best_match.reason,
            ),
            quality=_to_quality(result),
            template_preview_b64=template_preview,
            alignment_overlay_b64=overlay_b64,
            stages=result.stages_b64,
            vault_unlocked=vault_unlocked,
        )

    @application.get(
        "/api/users",
        response_model=list[UserSummary],
        tags=["Users"],
        summary="List all enrolled users",
    )
    async def list_users() -> list[UserSummary]:
        return [_to_user_summary(rec) for rec in template_store.list_users()]

    @application.get(
        "/api/users/{user_id}/template",
        tags=["Users"],
        summary="Download stored 1-px skeleton PNG for a user",
    )
    async def get_user_template(user_id: str) -> Response:
        try:
            data = template_store.get_template_bytes(user_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if data is None:
            raise HTTPException(status_code=404, detail="User template not found.")
        return Response(content=data, media_type="image/png")

    @application.delete(
        "/api/users/{user_id}",
        tags=["Users"],
        summary="Delete an enrolled user and skeleton template",
    )
    async def delete_user(user_id: str) -> dict[str, object]:
        try:
            deleted = template_store.delete_user(user_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="User not found.")
        return {"deleted": True, "user_id": user_id}

    @application.post(
        "/api/vault/session",
        response_model=VaultStatusResponse,
        tags=["Vault Unlocker"],
        summary="Create a locked Cross-Device Vault session",
    )
    async def create_vault_session() -> VaultStatusResponse:
        s = vault_store.create_session()
        return VaultStatusResponse(
            session_id=s.session_id,
            status=s.status,
            created_at=s.created_at,
        )

    @application.get(
        "/api/vault/status/{session_id}",
        response_model=VaultStatusResponse,
        tags=["Vault Unlocker"],
        summary="Poll Vault session lock status",
    )
    async def get_vault_status(session_id: str) -> VaultStatusResponse:
        s = vault_store.get_session(session_id)
        if s is None:
            raise HTTPException(status_code=404, detail="Vault session not found.")
        return VaultStatusResponse(
            session_id=s.session_id,
            status=s.status,
            created_at=s.created_at,
            unlocked_by=s.unlocked_by,
            score_percent=round(s.score * 100.0, 2) if s.score is not None else None,
            unlocked_at=s.unlocked_at,
        )

    @application.post(
        "/api/vault/lock/{session_id}",
        response_model=VaultStatusResponse,
        tags=["Vault Unlocker"],
        summary="Re-lock an active Vault session",
    )
    async def lock_vault_session(session_id: str) -> VaultStatusResponse:
        s = vault_store.lock_session(session_id)
        if s is None:
            raise HTTPException(status_code=404, detail="Vault session not found.")
        return VaultStatusResponse(
            session_id=s.session_id,
            status=s.status,
            created_at=s.created_at,
        )

    return application


app = create_app()
