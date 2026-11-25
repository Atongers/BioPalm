"""FastAPI application and OpenAPI endpoints for BioPalm Auth."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.net import get_lan_ip
from app.cv.matcher import (
    MatcherMode,
    MatchResult,
    compute_match_score,
    render_alignment_overlay,
    render_orb_matches,
)
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
from app.storage.templates import (
    EnrollMode,
    TemplateSample,
    TemplateStore,
    UserRecord,
    VaultSessionStore,
)


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
        alignment_method=res.alignment_method,
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
        sample_count=rec.sample_count,
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

    @application.get("/api/network-info", tags=["System"])
    async def network_info(request: Request) -> dict[str, object]:
        lan_ip = get_lan_ip()
        scheme = request.url.scheme or "https"
        port = request.url.port
        if port is None:
            port = 443 if scheme == "https" else 80
        mobile_url = f"{scheme}://{lan_ip}:{port}/"
        return {
            "lan_ip": lan_ip,
            "scheme": scheme,
            "port": port,
            "mobile_url": mobile_url,
        }

    @application.post(
        "/api/pipeline/inspect",
        response_model=InspectResponse,
        tags=["CV Pipeline"],
        summary="Preview 6-stage CV pipeline without storing template",
    )
    async def inspect_pipeline(
        file: Annotated[UploadFile, File(description="Palm image (PNG/JPEG)")],
        crop_mode: Annotated[CropMode, Form()] = "auto_palm",
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
        summary="Enroll a user's palmprint template (replace or append up to 3 samples)",
    )
    async def enroll_user(
        user_id: Annotated[str, Form(description="Alphanumeric ID, e.g. 'fathan'")],
        file: Annotated[UploadFile, File(description="Palm image (PNG/JPEG)")],
        display_name: Annotated[str, Form()] = "",
        enroll_mode: Annotated[EnrollMode, Form()] = "replace",
        crop_mode: Annotated[CropMode, Form()] = "auto_palm",
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
                enhanced_gray=result.enhanced_gray,
                mode=enroll_mode,
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
        "/api/enroll/multi",
        response_model=EnrollResponse,
        tags=["Biometric Auth"],
        summary="Batch-enroll 1 to 3 palm samples for a user in a single request",
    )
    async def enroll_user_multi(
        user_id: Annotated[str, Form(description="Alphanumeric ID, e.g. 'fathan'")],
        files: Annotated[list[UploadFile], File(description="1 to 3 palm images (PNG/JPEG)")],
        display_name: Annotated[str, Form()] = "",
        crop_mode: Annotated[CropMode, Form()] = "auto_palm",
        use_blackhat: Annotated[bool, Form()] = config.DEFAULT_USE_BLACKHAT,
        clip_limit: Annotated[float, Form()] = config.CLAHE_CLIP_LIMIT,
        median_ksize: Annotated[int, Form()] = config.MEDIAN_KSIZE,
        block_size: Annotated[int, Form()] = config.ADAPTIVE_BLOCK_SIZE,
        c_val: Annotated[int, Form()] = config.ADAPTIVE_C,
        min_component_px: Annotated[int, Form(ge=0, le=200)] = config.MIN_COMPONENT_PIXELS,
        include_stages: Annotated[bool, Form()] = True,
    ) -> EnrollResponse:
        if not files or len(files) > config.MAX_ENROLL_SAMPLES:
            raise HTTPException(
                status_code=400,
                detail=f"Provide between 1 and {config.MAX_ENROLL_SAMPLES} palm images.",
            )

        last_result: PipelineResult | None = None
        record: UserRecord | None = None
        for idx, upload in enumerate(files):
            image_bytes = await upload.read()
            result = _run_pipeline(
                image_bytes=image_bytes,
                crop_mode=crop_mode,
                roi_box=None,
                use_blackhat=use_blackhat,
                clip_limit=clip_limit,
                median_ksize=median_ksize,
                block_size=block_size,
                c_val=c_val,
                min_component_px=min_component_px,
                include_stages=include_stages and (idx == len(files) - 1),
            )
            if result.skeleton_density < config.MIN_SKELETON_RATIO:
                raise HTTPException(
                    status_code=422,
                    detail=f"Sample #{idx + 1} has insufficient palm features.",
                )
            try:
                record = template_store.save_template(
                    user_id=user_id,
                    display_name=display_name or user_id,
                    skeleton=result.skeleton,
                    enhanced_gray=result.enhanced_gray,
                    mode="replace" if idx == 0 else "append",
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            last_result = result

        assert record is not None and last_result is not None
        return EnrollResponse(
            enrolled=True,
            user=_to_user_summary(record),
            quality=_to_quality(last_result),
            stages=last_result.stages_b64,
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
        matcher_mode: Annotated[MatcherMode, Form()] = config.DEFAULT_MATCHER_MODE,  # type: ignore[assignment]
        dilation_ksize: Annotated[int, Form(ge=1, le=21)] = config.DEFAULT_DILATION_KSIZE,
        shift_range: Annotated[int, Form(ge=0, le=30)] = config.DEFAULT_SHIFT_RANGE,
        enable_alignment: Annotated[bool, Form()] = True,
        crop_mode: Annotated[CropMode, Form()] = "auto_palm",
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
        candidates: list[tuple[UserRecord, list[TemplateSample]]] = []
        if user_id and user_id.strip():
            try:
                rec = template_store.get_user_record(user_id.strip())
                samples = template_store.get_user_samples(user_id.strip())
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if rec is None or not samples:
                raise HTTPException(
                    status_code=404,
                    detail=f"Enrolled user '{user_id}' not found.",
                )
            candidates.append((rec, samples))
        else:
            for rec in template_store.list_users():
                samples = template_store.get_user_samples(rec.user_id)
                if samples:
                    candidates.append((rec, samples))
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
        best_sample: TemplateSample | None = None
        best_match: MatchResult | None = None
        total_samples_checked = 0

        for rec, samples in candidates:
            for sample in samples:
                total_samples_checked += 1
                m = compute_match_score(
                    probe_skeleton=result.skeleton,
                    template_skeleton=sample.skeleton,
                    probe_gray=result.enhanced_gray,
                    template_gray=sample.enhanced_gray,
                    matcher_mode=matcher_mode,
                    threshold=threshold,
                    dilation_ksize=dilation_ksize,
                    shift_range=shift_range,
                    enable_alignment=enable_alignment,
                )
                if best_match is None or m.score > best_match.score:
                    best_match = m
                    best_record = rec
                    best_sample = sample

        assert best_match is not None and best_record is not None and best_sample is not None

        vault_unlocked = False
        if best_match.verified and session_id and session_id.strip():
            unlocked = vault_store.unlock_session(
                session_id=session_id.strip(),
                user_id=best_record.user_id,
                score=best_match.score,
            )
            vault_unlocked = unlocked is not None

        template_preview = (
            encode_png_b64(best_sample.skeleton)
            if include_stages
            else None
        )
        overlay_b64 = None
        orb_b64 = None
        if include_stages:
            overlay_img = render_alignment_overlay(
                probe_skeleton=result.skeleton,
                template_skeleton=best_sample.skeleton,
                shift_x=best_match.shift_x,
                shift_y=best_match.shift_y,
                dilation_ksize=dilation_ksize,
            )
            overlay_b64 = encode_png_b64(overlay_img)
            if best_sample.enhanced_gray is not None:
                orb_img = render_orb_matches(result.enhanced_gray, best_sample.enhanced_gray)
                orb_b64 = encode_png_b64(orb_img)

        return VerifyResponse(
            verified=best_match.verified,
            status="TERVERIFIKASI" if best_match.verified else "DITOLAK",
            matched_user=_to_user_summary(best_record) if best_match.verified else None,
            match=MatchBreakdown(
                score=best_match.score,
                score_percent=round(best_match.score * 100.0, 2),
                matcher_mode=best_match.matcher_mode,
                dilated_iou=best_match.dilated_iou,
                dice_f1=best_match.dice_f1,
                hamming_similarity=best_match.hamming_similarity,
                orb_score=best_match.orb_score,
                orb_good_matches=best_match.orb_good_matches,
                threshold=threshold,
                dilation_ksize=dilation_ksize,
                shift_x=best_match.shift_x,
                shift_y=best_match.shift_y,
                matched_sample_index=best_sample.sample_index,
                samples_checked=total_samples_checked,
                reason=best_match.reason,
            ),
            quality=_to_quality(result),
            template_preview_b64=template_preview,
            alignment_overlay_b64=overlay_b64,
            orb_matches_b64=orb_b64,
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
        summary="Download stored 1-px skeleton PNG for a user sample",
    )
    async def get_user_template(user_id: str, sample_index: int = 0) -> Response:
        try:
            data = template_store.get_template_bytes(user_id, sample_index=sample_index)
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
