"""Gera CA local + certificado HTTPS assinado para desenvolvimento.

Saída padrão:
- certs/ca.crt / certs/ca.key          (autoridade local)
- certs/server.crt / certs/server.key  (certificado do servidor)

Browsers só param de avisar se a CA local estiver no trust store.
No Windows (Chrome/Edge):

    python scripts/generate_dev_https_cert.py --force --install-ca

Exemplo:
    python scripts/generate_dev_https_cert.py --force
"""
from __future__ import annotations

import argparse
import ipaddress
import socket
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Gera/renova CA local e certificado HTTPS assinado "
            "para desenvolvimento (com SAN/IP e extensões TLS modernas)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="certs",
        help="Diretório de saída dos arquivos PEM (padrão: certs).",
    )
    parser.add_argument(
        "--cert-file",
        default="server.crt",
        help="Nome do arquivo de certificado do servidor (padrão: server.crt).",
    )
    parser.add_argument(
        "--key-file",
        default="server.key",
        help="Nome do arquivo de chave privada do servidor (padrão: server.key).",
    )
    parser.add_argument(
        "--ca-cert-file",
        default="ca.crt",
        help="Nome do arquivo de certificado da CA (padrão: ca.crt).",
    )
    parser.add_argument(
        "--ca-key-file",
        default="ca.key",
        help="Nome do arquivo de chave privada da CA (padrão: ca.key).",
    )
    parser.add_argument(
        "--common-name",
        default="localhost",
        help="Common Name (CN) do certificado do servidor (padrão: localhost).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=825,
        help="Validade do certificado do servidor em dias (padrão: 825).",
    )
    parser.add_argument(
        "--ca-days",
        type=int,
        default=3650,
        help="Validade da CA em dias (padrão: 3650).",
    )
    parser.add_argument(
        "--key-size",
        type=int,
        default=4096,
        choices=(2048, 3072, 4096),
        help="Tamanho da chave RSA em bits (padrão: 4096).",
    )
    parser.add_argument(
        "--san",
        action="append",
        default=[],
        help=(
            "SAN extra (DNS ou IP). Pode repetir. "
            "Padrão já inclui localhost, 127.0.0.1 e ::1."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Sobrescreve arquivos existentes.",
    )
    parser.add_argument(
        "--install-ca",
        action="store_true",
        help=(
            "Instala certs/ca.crt no trust store do usuário Windows "
            "(Chrome/Edge). Requer PowerShell/certutil."
        ),
    )
    parser.add_argument(
        "--reuse-ca",
        action="store_true",
        help="Reutiliza CA existente em vez de regenerá-la (exige ca.crt/ca.key).",
    )
    return parser


def _org_name(common_name: str, *, is_ca: bool):
    from cryptography import x509
    from cryptography.x509.oid import NameOID

    cn = "IMeC Analysis Local CA" if is_ca else common_name
    return x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "BR"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "SP"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "Sao Paulo"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "IMeC Analysis Dev"),
            x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Local TLS"),
            x509.NameAttribute(NameOID.COMMON_NAME, cn),
        ]
    )


def _build_san_entries(common_name: str, extra_sans: list[str]) -> list:
    from cryptography import x509

    entries: list = [
        x509.DNSName("localhost"),
        x509.DNSName("*.localhost"),
        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        x509.IPAddress(ipaddress.IPv6Address("::1")),
    ]
    if common_name and common_name.lower() != "localhost":
        entries.append(x509.DNSName(common_name))

    try:
        hostname = socket.gethostname().strip()
        if hostname:
            entries.append(x509.DNSName(hostname))
            fqdn = socket.getfqdn().strip()
            if fqdn and fqdn != hostname:
                entries.append(x509.DNSName(fqdn))
    except OSError:
        pass

    for raw in extra_sans:
        value = raw.strip()
        if not value:
            continue
        try:
            entries.append(x509.IPAddress(ipaddress.ip_address(value)))
        except ValueError:
            entries.append(x509.DNSName(value))

    # Dedup preservando ordem
    seen: set[str] = set()
    unique = []
    for entry in entries:
        key = entry.__repr__()
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    return unique


def _write_key(path: Path, key) -> None:
    from cryptography.hazmat.primitives import serialization

    path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )


def _write_cert(path: Path, certificate) -> None:
    from cryptography.hazmat.primitives import serialization

    path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))


def _load_or_create_ca(
    *,
    ca_cert_path: Path,
    ca_key_path: Path,
    key_size: int,
    ca_days: int,
    reuse_ca: bool,
    force: bool,
):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    if reuse_ca or (not force and ca_cert_path.exists() and ca_key_path.exists()):
        if not ca_cert_path.exists() or not ca_key_path.exists():
            raise FileNotFoundError(
                f"CA não encontrada para reuso: {ca_cert_path} / {ca_key_path}"
            )
        ca_cert = x509.load_pem_x509_certificate(ca_cert_path.read_bytes())
        ca_key = serialization.load_pem_private_key(ca_key_path.read_bytes(), password=None)
        return ca_cert, ca_key, False

    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)
    subject = _org_name("IMeC Analysis Local CA", is_ca=True)
    now = datetime.now(timezone.utc)
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=ca_days))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_cert_sign=True,
                crl_sign=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()),
            critical=False,
        )
        .sign(private_key=ca_key, algorithm=hashes.SHA256())
    )
    return ca_cert, ca_key, True


def _build_server_cert(
    *,
    ca_cert,
    ca_key,
    common_name: str,
    days: int,
    key_size: int,
    extra_sans: list[str],
):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import ExtendedKeyUsageOID

    server_key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)
    subject = _org_name(common_name, is_ca=False)
    now = datetime.now(timezone.utc)
    san = x509.SubjectAlternativeName(_build_san_entries(common_name, extra_sans))

    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_cert.subject)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_encipherment=True,
                key_agreement=False,
                content_commitment=False,
                data_encipherment=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .add_extension(san, critical=False)
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(server_key.public_key()),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
            critical=False,
        )
    )
    certificate = builder.sign(private_key=ca_key, algorithm=hashes.SHA256())
    return certificate, server_key


def _install_ca_windows(ca_cert_path: Path) -> int:
    """Instala a CA no store Root do usuário atual (Chrome/Edge)."""
    if sys.platform != "win32":
        print(
            "Aviso: --install-ca automático só está implementado no Windows. "
            f"Importe manualmente: {ca_cert_path}",
            file=sys.stderr,
        )
        return 1

    # Remove entrada anterior com o mesmo subject (best-effort) e reinstala.
    cmd = [
        "certutil",
        "-user",
        "-addstore",
        "Root",
        str(ca_cert_path.resolve()),
    ]
    print("Instalando CA no trust store do usuário (Root)...")
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        print(
            "Falha ao instalar a CA. Execute o PowerShell como o usuário da sessão "
            "e rode de novo com --install-ca, ou importe certs/ca.crt manualmente "
            "em 'Autoridades de Certificação Raiz Confiáveis' (usuário atual).",
            file=sys.stderr,
        )
        return completed.returncode

    print(completed.stdout.strip() or "CA instalada com sucesso.")
    print(
        "Reinicie o Chrome/Edge se a aba já estava aberta. "
        "Firefox usa trust store próprio: importe certs/ca.crt em "
        "Configurações > Privacidade e Segurança > Certificados > Autoridades."
    )
    return 0


def main() -> int:
    args = _build_parser().parse_args()

    if args.days < 1 or args.ca_days < 1:
        print("Erro: --days e --ca-days devem ser >= 1.", file=sys.stderr)
        return 2

    try:
        from cryptography import x509  # noqa: F401
        from cryptography.hazmat.primitives import hashes, serialization  # noqa: F401
        from cryptography.hazmat.primitives.asymmetric import rsa  # noqa: F401
        from cryptography.x509.oid import NameOID  # noqa: F401
    except ImportError:
        print(
            "Dependência ausente: cryptography. Instale com "
            "'python -m pip install -r requirements.txt'.",
            file=sys.stderr,
        )
        return 2

    output_dir = Path(args.output_dir)
    cert_path = output_dir / args.cert_file
    key_path = output_dir / args.key_file
    ca_cert_path = output_dir / args.ca_cert_file
    ca_key_path = output_dir / args.ca_key_file

    # Apenas instalar CA já existente, sem regenerar.
    if args.install_ca and not args.force and cert_path.exists() and ca_cert_path.exists():
        if not key_path.exists():
            print(f"Aviso: chave do servidor ausente em {key_path}", file=sys.stderr)
        return _install_ca_windows(ca_cert_path)

    server_exists = cert_path.exists() or key_path.exists()
    if not args.force and server_exists:
        print(
            "Arquivos do servidor já existem. Use --force para sobrescrever: "
            f"{cert_path} e {key_path}",
            file=sys.stderr,
        )
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        ca_cert, ca_key, ca_created = _load_or_create_ca(
            ca_cert_path=ca_cert_path,
            ca_key_path=ca_key_path,
            key_size=args.key_size,
            ca_days=args.ca_days,
            reuse_ca=args.reuse_ca,
            force=args.force and not args.reuse_ca,
        )
    except FileNotFoundError as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 2

    if ca_created:
        _write_key(ca_key_path, ca_key)
        _write_cert(ca_cert_path, ca_cert)

    server_cert, server_key = _build_server_cert(
        ca_cert=ca_cert,
        ca_key=ca_key,
        common_name=args.common_name,
        days=args.days,
        key_size=args.key_size,
        extra_sans=args.san,
    )
    _write_key(key_path, server_key)
    _write_cert(cert_path, server_cert)

    from cryptography import x509 as _x509

    san_ext = server_cert.extensions.get_extension_for_class(_x509.SubjectAlternativeName)
    san_text = ", ".join(str(name.value) for name in san_ext.value)

    print(f"CA: {'criada' if ca_created else 'reutilizada'} -> {ca_cert_path}")
    print(f"Certificado do servidor: {cert_path}")
    print(f"Chave do servidor: {key_path}")
    print(f"Algoritmo: RSA-{args.key_size} / SHA-256")
    print(f"SAN: {san_text}")
    print(
        "Para o browser confiar neste certificado, instale a CA local:\n"
        "  python scripts/generate_dev_https_cert.py --install-ca\n"
        f"  (ou importe {ca_cert_path} em Autoridades Raiz Confiáveis)"
    )

    if args.install_ca:
        return _install_ca_windows(ca_cert_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
