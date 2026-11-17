"""API & OpenAPI integration tests for BioPalm Auth."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app
from tests.test_pipeline import make_blank_image, make_synthetic_palm


def _client(tmp_path) -> TestClient:
    app = create_app(data_dir=tmp_path)
    return TestClient(app)


def test_openapi_docs_and_web_ui_available(tmp_path):
    client = _client(tmp_path)

    openapi_res = client.get("/openapi.json")
    assert openapi_res.status_code == 200
    paths = openapi_res.json()["paths"]
    assert "/api/enroll" in paths
    assert "/api/verify" in paths
    assert "/api/pipeline/inspect" in paths
    assert "/api/vault/session" in paths

    assert client.get("/docs").status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/health").json()["status"] == "ok"


def test_inspect_enroll_verify_and_vault_flow(tmp_path):
    client = _client(tmp_path)
    palm_a = make_synthetic_palm("A")
    palm_b = make_synthetic_palm("B")
    blank = make_blank_image()

    # 1. Inspect 6 stages
    inspect_res = client.post(
        "/api/pipeline/inspect",
        files={"file": ("palm.png", palm_a, "image/png")},
        data={"crop_mode": "full"},
    )
    assert inspect_res.status_code == 200
    assert "stage5b_skeleton_1px" in inspect_res.json()["stages"]

    # 2. Blank image should be rejected on enroll
    blank_res = client.post(
        "/api/enroll",
        files={"file": ("blank.png", blank, "image/png")},
        data={"user_id": "empty_user", "crop_mode": "full"},
    )
    assert blank_res.status_code == 422

    # 3. Enroll user
    enroll_res = client.post(
        "/api/enroll",
        files={"file": ("palm_a.png", palm_a, "image/png")},
        data={"user_id": "fathan", "display_name": "Fathan", "crop_mode": "full"},
    )
    assert enroll_res.status_code == 200
    assert enroll_res.json()["enrolled"] is True

    # List & download template
    users = client.get("/api/users").json()
    assert len(users) == 1
    assert users[0]["user_id"] == "fathan"
    assert client.get("/api/users/fathan/template").status_code == 200

    # 4. Create Vault session (locked)
    vault_res = client.post("/api/vault/session")
    assert vault_res.status_code == 200
    sid = vault_res.json()["session_id"]
    assert vault_res.json()["status"] == "locked"

    # 5. Verify with wrong palm -> DITOLAK, Vault remains locked
    fail_res = client.post(
        "/api/verify",
        files={"file": ("palm_b.png", palm_b, "image/png")},
        data={"session_id": sid, "crop_mode": "full"},
    )
    assert fail_res.status_code == 200
    assert fail_res.json()["verified"] is False
    assert fail_res.json()["status"] == "DITOLAK"
    assert fail_res.json()["vault_unlocked"] is False
    assert client.get(f"/api/vault/status/{sid}").json()["status"] == "locked"

    # 6. Verify 1:N with shifted matching palm -> TERVERIFIKASI & Vault unlocked
    ok_res = client.post(
        "/api/verify",
        files={"file": ("palm_a2.png", make_synthetic_palm("A", shift_x=6, shift_y=-4), "image/png")},
        data={"session_id": sid, "crop_mode": "full"},
    )
    assert ok_res.status_code == 200
    body = ok_res.json()
    assert body["verified"] is True
    assert body["status"] == "TERVERIFIKASI"
    assert body["matched_user"]["user_id"] == "fathan"
    assert body["vault_unlocked"] is True
    assert body["alignment_overlay_b64"].startswith("data:image/png;base64,")
    assert "shift_x" in body["match"] and "shift_y" in body["match"]

    vault_status = client.get(f"/api/vault/status/{sid}").json()
    assert vault_status["status"] == "unlocked"
    assert vault_status["unlocked_by"] == "fathan"

    # Re-lock Vault and delete user
    assert client.post(f"/api/vault/lock/{sid}").json()["status"] == "locked"
    assert client.delete("/api/users/fathan").json()["deleted"] is True
    assert client.get("/api/users").json() == []
