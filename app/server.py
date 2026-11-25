"""HTTPS LAN launcher for BioPalm Auth so phones can access getUserMedia directly."""

from __future__ import annotations

import argparse

import uvicorn

from app import config
from app.net import ensure_self_signed_cert


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BioPalm Auth over HTTPS on LAN.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()

    cert_dir = config.BASE_DIR / "certs"
    cert_path, key_path, lan_ip = ensure_self_signed_cert(cert_dir)

    print(f"BioPalm HTTPS ready:")
    print(f"  - Laptop : https://localhost:{args.port}/")
    print(f"  - Mobile : https://{lan_ip}:{args.port}/")
    print(f"  - OpenAPI: https://{lan_ip}:{args.port}/docs")

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        ssl_certfile=str(cert_path),
        ssl_keyfile=str(key_path),
    )


if __name__ == "__main__":
    main()
