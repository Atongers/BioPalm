"""Local file template store and Vault session manager."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import secrets

import cv2
import numpy as np

from app import config

USER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


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
class UserRecord:
    """Metadata stored for an enrolled user."""

    user_id: str
    display_name: str
    skeleton_pixels: int
    created_at: str
    updated_at: str


class TemplateStore:
    """Persists 256x256 binary palm skeletons as PNGs plus JSON user metadata."""

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

    def _template_path(self, user_id: str) -> Path:
        return self.templates_dir / f"{user_id}.png"

    def save_template(
        self,
        user_id: str,
        display_name: str,
        skeleton: np.ndarray,
    ) -> UserRecord:
        uid = validate_user_id(user_id)
        name = display_name.strip() or uid
        path = self._template_path(uid)

        ok = cv2.imwrite(str(path), skeleton)
        if not ok:
            raise RuntimeError(f"Failed to write template image for user '{uid}'.")

        users = self._read_users()
        now = _utc_now_iso()
        existing = users.get(uid)
        created_at = existing["created_at"] if existing else now
        record = UserRecord(
            user_id=uid,
            display_name=name,
            skeleton_pixels=int(np.count_nonzero(skeleton)),
            created_at=created_at,
            updated_at=now,
        )
        users[uid] = asdict(record)
        self._write_users(users)
        return record

    def get_template(self, user_id: str) -> tuple[UserRecord, np.ndarray] | None:
        uid = validate_user_id(user_id)
        users = self._read_users()
        raw = users.get(uid)
        path = self._template_path(uid)
        if raw is None or not path.exists():
            return None

        skeleton = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if skeleton is None:
            return None
        return UserRecord(**raw), skeleton

    def get_template_bytes(self, user_id: str) -> bytes | None:
        uid = validate_user_id(user_id)
        path = self._template_path(uid)
        if not path.exists():
            return None
        return path.read_bytes()

    def list_users(self) -> list[UserRecord]:
        users = self._read_users()
        return [UserRecord(**item) for item in users.values()]

    def delete_user(self, user_id: str) -> bool:
        uid = validate_user_id(user_id)
        users = self._read_users()
        existed = uid in users
        if existed:
            del users[uid]
            self._write_users(users)
        path = self._template_path(uid)
        if path.exists():
            path.unlink()
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
