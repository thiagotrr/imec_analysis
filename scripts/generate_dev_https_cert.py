"""Gera certificado HTTPS autoassinado para ambiente de desenvolvimento.

Saída padrão:
- certs/server.crt
- certs/server.key

Exemplo:
    python scripts/generate_dev_https_cert.py
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gera/renova certificado HTTPS autoassinado para desenvolvimento.",
    )
    parser.add_argument(
        "--output-dir",
        default="certs",
        help="Diretório de saída dos arquivos PEM (padrão: certs).",
    )
    parser.add_argument(
        "--cert-file",
        default="server.crt",
        help="Nome do arquivo de certificado PEM (padrão: server.crt).",
    )
    parser.add_argument(
        "--key-file",
        default="server.key",
        help="Nome do arquivo de chave privada PEM (padrão: server.key).",
    )
    parser.add_argument(
        "--common-name",
        default="localhost",
        help="Common Name (CN) do certificado (padrão: localhost).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Validade em dias (padrão: 365).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Sobrescreve arquivos existentes.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    if args.days < 1:
        print("Erro: --days deve ser >= 1.", file=sys.stderr)
        return 2

    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        print(
            "Dependência ausente: cryptography. Instale com 'python -m pip install -r requirements.txt'.",
            file=sys.stderr,
        )
        return 2

    output_dir = Path(args.output_dir)
    cert_path = output_dir / args.cert_file
    key_path = output_dir / args.key_file

    if not args.force and (cert_path.exists() or key_path.exists()):
        print(
            "Arquivos já existem. Use --force para sobrescrever: "
            f"{cert_path} e {key_path}",
            file=sys.stderr,
        )
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "BR"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "SP"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "Sao Paulo"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "IMeC Analysis Dev"),
            x509.NameAttribute(NameOID.COMMON_NAME, args.common_name),
        ]
    )

    now = datetime.now(timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=args.days))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(args.common_name)]),
            critical=False,
        )
        .sign(private_key=key, algorithm=hashes.SHA256())
    )

    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))

    print(f"Certificado criado: {cert_path}")
    print(f"Chave criada: {key_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
