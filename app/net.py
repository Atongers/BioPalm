"""LAN IP discovery and local self-signed TLS certificate generation for mobile HTTPS."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import ipaddress
from pathlib import Path
import socket

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


def get_lan_ip() -> str:
    """Detect the active LAN IPv4 address used for outbound Wi-Fi/Ethernet traffic."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        if ip and not ip.startswith("127."):
            return ip
    except OSError:
        pass
    finally:
        sock.close()

    try:
        hostname_ip = socket.gethostbyname(socket.gethostname())
        if hostname_ip and not hostname_ip.startswith("127."):
            return hostname_ip
    except OSError:
        pass
    return "127.0.0.1"


def ensure_self_signed_cert(
    cert_dir: Path,
    lan_ip: str | None = None,
) -> tuple[Path, Path, str]:
    """Create or reuse a self-signed TLS cert with SANs for localhost, 127.0.0.1, and LAN IP."""
    resolved_ip = lan_ip or get_lan_ip()
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert_path = cert_dir / "cert.pem"
    key_path = cert_dir / "key.pem"
    ip_marker = cert_dir / "lan_ip.txt"

    if (
        cert_path.exists()
        and key_path.exists()
        and ip_marker.exists()
        and ip_marker.read_text(encoding="utf-8").strip() == resolved_ip
    ):
        return cert_path, key_path, resolved_ip

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "BioPalm Local Prototype")]
    )

    san_entries: list[x509.GeneralName] = [
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
    ]
    try:
        ip_obj = ipaddress.ip_address(resolved_ip)
        if str(ip_obj) != "127.0.0.1":
            san_entries.append(x509.IPAddress(ip_obj))
    except ValueError:
        pass

    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .sign(key, hashes.SHA256())
    )

    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    ip_marker.write_text(resolved_ip, encoding="utf-8")
    return cert_path, key_path, resolved_ip
