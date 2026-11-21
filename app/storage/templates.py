"""Local file multi-template store and Vault session manager."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import secrets
import shutil
from typing import Literal

import cv2
import numpy as np

from app import config

USER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
EnrollMode = Literal["replace", "append"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_user_id(user_id: str) -> str:
    """Validate user_id to prevent invalid filenames or path traversal."""
    cleaned = user_id.strip()
    if not USER_ID_PATTERN.match(cleaned):
        raise ValueError(
            "user_id must be 1-64 characters using only letters, digits, '_' or '-'."
        )
    return cleaned


@dataclass
class TemplateSample:
    """In-memory representation of one stored biometric sample."""

    sample_index: int
    skeleton: np.ndarray
    enhanced_gray: np.ndarray | None = None


@dataclass
class UserRecord:
    """Metadata stored for an enrolled user."""

    user_id: str
    display_name: str
    skeleton_pixels: int
    created_at: str
    updated_at: str
    sample_count: int = 1
    samples: list[dict] | None = None


class TemplateStore:
    """Persists up to 3 256x256 palm samples (skeleton + CLAHE gray) per user."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or config.DATA_DIR
        self.templates_dir = self.data_dir / "templates"
        self.users_file = self.data_dir / "users.json"
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        if not self.users_file.exists():
            self._write_users({})

    def _read_users(self) -> dict[str, dict]:
        with self.users_file.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _write_users(self, users: dict[str, dict]) -> None:
        with self.users_file.open("w", encoding="utf-8") as fh:
            json.dump(users, fh, indent=2)

    def _user_dir(self, user_id: str) -> Path:
        return self.templates_dir / user_id

    def _legacy_template_path(self, user_id: str) -> Path:
        return self.templates_dir / f"{user_id}.png"

    def _sample_skel_path(self, user_id: str, idx: int) -> Path:
        return self._user_dir(user_id) / f"sample_{idx}_skel.png"

    def _sample_gray_path(self, user_id: str, idx: int) -> Path:
        return self._user_dir(user_id) / f"sample_{idx}_gray.png"

    def save_template(
        self,
        user_id: str,
        display_name: str,
        skeleton: np.ndarray,
        enhanced_gray: np.ndarray | None = None,
        mode: EnrollMode = "replace",
    ) -> UserRecord:
        uid = validate_user_id(user_id)
        name = display_name.strip() or uid
        user_dir = self._user_dir(uid)

        users = self._read_users()
        existing = users.get(uid)
        now = _utc_now_iso()

        if mode == "replace" or existing is None:
            if user_dir.exists():
                shutil.rmtree(user_dir, ignore_errors=True)
            legacy = self._legacy_template_path(uid)
            if legacy.exists():
                legacy.unlink()
            samples_meta: list[dict] = []
            sample_idx = 0
            created_at = existing["created_at"] if existing else now
        elif mode == "append":
            samples_meta = list(existing.get("samples") or [])
            if not samples_meta and self._legacy_template_path(uid).exists():
                # Migrate legacy single-sample layout to folder layout before appending
                legacy_skel = cv2.imread(
                    str(self._legacy_template_path(uid)), cv2.IMREAD_GRAYSCALE
                )
                if legacy_skel is not None:
                    user_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(self._sample_skel_path(uid, 0)), legacy_skel)
                    self._legacy_template_path(uid).unlink()
                    samples_meta.append(
                        {
                            "sample_index": 0,
                            "skeleton_pixels": int(np.count_nonzero(legacy_skel)),
                            "created_at": existing["created_at"],
                        }
                    )
            if len(samples_meta) >= config.MAX_ENROLL_SAMPLES:
                raise ValueError(
                    f"Maximum {config.MAX_ENROLL_SAMPLES} samples reached for user '{uid}'. "
                    "Use enroll_mode='replace' to reset."
                )
            sample_idx = len(samples_meta)
            created_at = existing["created_at"]
        else:
            raise ValueError(f"Unsupported enroll_mode: {mode}")

        user_dir.mkdir(parents=True, exist_ok=True)
        skel_path = self._sample_skel_path(uid, sample_idx)
        if not cv2.imwrite(str(skel_path), skeleton):
            raise RuntimeError(f"Failed to write skeleton sample for user '{uid}'.")

        if enhanced_gray is not None:
            gray_path = self._sample_gray_path(uid, sample_idx)
            if not cv2.imwrite(str(gray_path), enhanced_gray):
                raise RuntimeError(f"Failed to write gray sample for user '{uid}'.")

        px_count = int(np.count_nonzero(skeleton))
        samples_meta.append(
            {
                "sample_index": sample_idx,
                "skeleton_pixels": px_count,
                "created_at": now,
            }
        )
        avg_pixels = int(
            round(sum(s["skeleton_pixels"] for s in samples_meta) / len(samples_meta))
        )

        record = UserRecord(
            user_id=uid,
            display_name=name,
            skeleton_pixels=avg_pixels,
            created_at=created_at,
            updated_at=now,
            sample_count=len(samples_meta),
            samples=samples_meta,
        )
        users[uid] = asdict(record)
        self._write_users(users)
        return record

    def get_user_record(self, user_id: str) -> UserRecord | None:
        uid = validate_user_id(user_id)
        users = self._read_users()
        raw = users.get(uid)
        if raw is None:
            return None
        return UserRecord(
            user_id=raw["user_id"],
            display_name=raw["display_name"],
            skeleton_pixels=raw["skeleton_pixels"],
            created_at=raw["created_at"],
            updated_at=raw["updated_at"],
            sample_count=raw.get("sample_count", 1),
            samples=raw.get("samples"),
        )

    def get_user_samples(self, user_id: str) -> list[TemplateSample]:
        uid = validate_user_id(user_id)
        record = self.get_user_record(uid)
        if record is None:
            return []

        loaded: list[TemplateSample] = []
        if record.samples:
            for item in record.samples:
                idx = int(item["sample_index"])
                skel_path = self._sample_skel_path(uid, idx)
                if not skel_path.exists():
                    continue
                skel = cv2.imread(str(skel_path), cv2.IMREAD_GRAYSCALE)
                if skel is None:
                    continue
                gray_path = self._sample_gray_path(uid, idx)
                gray = (
                    cv2.imread(str(gray_path), cv2.IMREAD_GRAYSCALE)
                    if gray_path.exists()
                    else None
                )
                loaded.append(
                    TemplateSample(
                        sample_index=idx,
                        skeleton=skel,
                        enhanced_gray=gray,
                    )
                )

        if not loaded and self._legacy_template_path(uid).exists():
            skel = cv2.imread(str(self._legacy_template_path(uid)), cv2.IMREAD_GRAYSCALE)
            if skel is not None:
                loaded.append(TemplateSample(sample_index=0, skeleton=skel, enhanced_gray=None))

        return loaded

    def get_template(self, user_id: str) -> tuple[UserRecord, np.ndarray] | None:
        """Backwards-compatible single-template getter (returns sample 0)."""
        rec = self.get_user_record(user_id)
        samples = self.get_user_samples(user_id)
        if rec is None or not samples:
            return None
        return rec, samples[0].skeleton

    def get_template_bytes(self, user_id: str, sample_index: int = 0) -> bytes | None:
        uid = validate_user_id(user_id)
        path = self._sample_skel_path(uid, sample_index)
        if path.exists():
            return path.read_bytes()
        if sample_index == 0 and self._legacy_template_path(uid).exists():
            return self._legacy_template_path(uid).read_bytes()
        return None

    def list_users(self) -> list[UserRecord]:
        users = self._read_users()
        out: list[UserRecord] = []
        for raw in users.values():
            out.append(
                UserRecord(
                    user_id=raw["user_id"],
                    display_name=raw["display_name"],
                    skeleton_pixels=raw["skeleton_pixels"],
                    created_at=raw["created_at"],
                    updated_at=raw["updated_at"],
                    sample_count=raw.get("sample_count", 1),
                    samples=raw.get("samples"),
                )
            )
        return out

    def delete_user(self, user_id: str) -> bool:
        uid = validate_user_id(user_id)
        users = self._read_users()
        existed = uid in users
        if existed:
            del users[uid]
            self._write_users(users)
        user_dir = self._user_dir(uid)
        if user_dir.exists():
            shutil.rmtree(user_dir, ignore_errors=True)
            existed = True
        legacy = self._legacy_template_path(uid)
        if legacy.exists():
            legacy.unlink()
            existed = True
        return existed


@dataclass
class VaultSession:
    """Cross-Device Web Unlocker session state."""

    session_id: str
    status: str  # "locked" | "unlocked"
    created_at: str
    unlocked_by: str | None = None
    score: float | None = None
    unlocked_at: str | None = None


class VaultSessionStore:
    """In-memory session store for Cross-Device Web Unlocker demo."""

    def __init__(self) -> None:
        self._sessions: dict[str, VaultSession] = {}

    def create_session(self) -> VaultSession:
        sid = secrets.token_urlsafe(6)
        session = VaultSession(
            session_id=sid,
            status="locked",
            created_at=_utc_now_iso(),
        )
        self._sessions[sid] = session
        return session

    def get_session(self, session_id: str) -> VaultSession | None:
        return self._sessions.get(session_id)

    def unlock_session(
        self, session_id: str, user_id: str, score: float
    ) -> VaultSession | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        session.status = "unlocked"
        session.unlocked_by = user_id
        session.score = score
        session.unlocked_at = _utc_now_iso()
        return session

    def lock_session(self, session_id: str) -> VaultSession | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        session.status = "locked"
        session.unlocked_by = None
        session.score = None
        session.unlocked_at = None
        return session
