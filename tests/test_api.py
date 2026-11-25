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

    net_res = client.get("/api/network-info")
    assert net_res.status_code == 200
    assert "mobile_url" in net_res.json()


def test_self_signed_cert_generation(tmp_path):
    from app.net import ensure_self_signed_cert

    cert_path, key_path, ip = ensure_self_signed_cert(tmp_path, lan_ip="192.168.1.50")
    assert cert_path.exists()
    assert key_path.exists()
    assert ip == "192.168.1.50"

    # Reuses existing files when LAN IP matches
    mtime_before = cert_path.stat().st_mtime_ns
    cert_path2, _, _ = ensure_self_signed_cert(tmp_path, lan_ip="192.168.1.50")
    assert cert_path2.stat().st_mtime_ns == mtime_before


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


def test_multi_sample_enrollment_append_and_batch(tmp_path):
    from tests.test_pipeline import make_synthetic_full_hand

    client = _client(tmp_path)
    s0 = make_synthetic_full_hand(0.0, 1.0)
    s1 = make_synthetic_full_hand(12.0, 0.95)
    s2 = make_synthetic_full_hand(-10.0, 0.85)

    # Sequential append enrollment up to 3 samples
    r1 = client.post(
        "/api/enroll",
        files={"file": ("s0.png", s0, "image/png")},
        data={"user_id": "multi_u", "enroll_mode": "replace", "crop_mode": "auto_palm"},
    )
    assert r1.status_code == 200
    assert r1.json()["user"]["sample_count"] == 1

    r2 = client.post(
        "/api/enroll",
        files={"file": ("s1.png", s1, "image/png")},
        data={"user_id": "multi_u", "enroll_mode": "append", "crop_mode": "auto_palm"},
    )
    assert r2.status_code == 200
    assert r2.json()["user"]["sample_count"] == 2

    r3 = client.post(
        "/api/enroll",
        files={"file": ("s2.png", s2, "image/png")},
        data={"user_id": "multi_u", "enroll_mode": "append", "crop_mode": "auto_palm"},
    )
    assert r3.status_code == 200
    assert r3.json()["user"]["sample_count"] == 3

    # 4th append should be rejected cleanly with HTTP 400
    r4 = client.post(
        "/api/enroll",
        files={"file": ("s3.png", s0, "image/png")},
        data={"user_id": "multi_u", "enroll_mode": "append", "crop_mode": "auto_palm"},
    )
    assert r4.status_code == 400

    # Verify picks highest score across all 3 stored samples (including hybrid + ORB visual)
    v_res = client.post(
        "/api/verify",
        files={"file": ("probe.png", make_synthetic_full_hand(13.0, 0.94), "image/png")},
        data={
            "user_id": "multi_u",
            "crop_mode": "auto_palm",
            "matcher_mode": "hybrid",
            "threshold": "0.75",
        },
    )
    assert v_res.status_code == 200
    v_body = v_res.json()
    assert v_body["verified"] is True
    assert v_body["match"]["samples_checked"] == 3
    assert v_body["match"]["matcher_mode"] == "hybrid"
    assert v_body["orb_matches_b64"].startswith("data:image/png;base64,")

    # Batch multi-enroll endpoint
    batch_res = client.post(
        "/api/enroll/multi",
        files=[
            ("files", ("b0.png", s0, "image/png")),
            ("files", ("b1.png", s1, "image/png")),
            ("files", ("b2.png", s2, "image/png")),
        ],
        data={"user_id": "batch_u", "display_name": "Batch User", "crop_mode": "auto_palm"},
    )
    assert batch_res.status_code == 200
    assert batch_res.json()["user"]["sample_count"] == 3
