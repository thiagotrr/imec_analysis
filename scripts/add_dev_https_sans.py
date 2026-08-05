"""Reemite o certificado HTTPS do servidor com IPs/DNS adicionais no SAN.

Reutiliza a CA local existente (`certs/ca.crt` / `certs/ca.key`), então máquinas
que já confiam na CA não precisam reinstalar — apenas o `server.crt`/`server.key`
são regenerados.

Exemplos:
    python scripts/add_dev_https_sans.py 192.168.68.104
    python scripts/add_dev_https_sans.py 192.168.68.104 10.0.0.5 api.local
    python scripts/add_dev_https_sans.py 192.168.68.104 --install-ca
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reemite certs/server.crt com SANs extras (IPs ou DNS), "
            "reutilizando a CA local."
        ),
    )
    parser.add_argument(
        "sans",
        nargs="+",
        metavar="SAN",
        help="IP ou nome DNS a incluir no certificado (um ou mais).",
    )
    parser.add_argument(
        "--output-dir",
        default="certs",
        help="Diretório dos arquivos PEM (padrão: certs).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=825,
        help="Validade do certificado do servidor em dias (padrão: 825).",
    )
    parser.add_argument(
        "--key-size",
        type=int,
        default=4096,
        choices=(2048, 3072, 4096),
        help="Tamanho da chave RSA em bits (padrão: 4096).",
    )
    parser.add_argument(
        "--common-name",
        default="localhost",
        help="Common Name do certificado (padrão: localhost).",
    )
    parser.add_argument(
        "--install-ca",
        action="store_true",
        help="Também instala/reinstala a CA no trust store do usuário Windows.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    generator = repo_root / "scripts" / "generate_dev_https_cert.py"

    if not generator.is_file():
        print(f"Erro: gerador não encontrado: {generator}", file=sys.stderr)
        return 2

    ca_cert = repo_root / args.output_dir / "ca.crt"
    ca_key = repo_root / args.output_dir / "ca.key"
    if not ca_cert.is_file() or not ca_key.is_file():
        print(
            "Erro: CA local ausente. Gere primeiro com:\n"
            "  python scripts/generate_dev_https_cert.py --force --install-ca",
            file=sys.stderr,
        )
        return 2

    cmd = [
        sys.executable,
        str(generator),
        "--force",
        "--reuse-ca",
        "--output-dir",
        args.output_dir,
        "--common-name",
        args.common_name,
        "--days",
        str(args.days),
        "--key-size",
        str(args.key_size),
    ]
    for san in args.sans:
        cmd.extend(["--san", san])
    if args.install_ca:
        cmd.append("--install-ca")

    print("Reemitindo certificado com SANs extras:", ", ".join(args.sans))
    completed = subprocess.run(cmd, cwd=str(repo_root), check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
