"""CLI administrativo de usuários de autenticação (Task 010).

Não existe endpoint HTTP de cadastro (decisão confirmada — ver
``docs/task010_*.md``): usuários são criados só por este script, que grava
diretamente na coleção ``users`` do Firestore via
``src/auth/users_repository.py``.

Uso:

    python scripts/db/firestore/gerenciar_usuarios.py create-user
    python scripts/db/firestore/gerenciar_usuarios.py load-initial

``create-user`` pede o e-mail (domínio obrigatório @energisa.com.br) e a
senha de forma interativa (senha não ecoa no terminal).

``load-initial`` faz carga inicial em lote: lê o arquivo
``scripts/db/firestore/usuarios.json`` (MESMO DIRETÓRIO deste script) e
cria/atualiza um usuário por entrada. Para popular esse arquivo, crie-o a
partir do exemplo abaixo (ele NUNCA deve ser commitado — está no
``.gitignore`` do repositório, pois contém senhas em texto claro):

    [
      {"email": "fulano@energisa.com.br", "senha": "TrocarNoPrimeiroLogin1!"},
      {"email": "ciclana@energisa.com.br", "senha": "TrocarNoPrimeiroLogin2!"}
    ]

Após a carga inicial, oriente os usuários a solicitar troca de senha (não há
endpoint de "trocar senha" nesta frente — refazer via ``create-user``,
que sobrescreve o hash existente).
"""
from __future__ import annotations

import argparse
import getpass
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[3] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from auth.security import hash_password  # noqa: E402
from auth.settings import ALLOWED_EMAIL_DOMAIN  # noqa: E402
from auth.users_repository import create_user, normalize_email  # noqa: E402
from db.firestore.client import load_firestore_runtime  # noqa: E402

USUARIOS_JSON_PATH = Path(__file__).resolve().parent / "usuarios.json"


def _validate_email(email: str) -> str:
    normalized = normalize_email(email)
    if not normalized.endswith(ALLOWED_EMAIL_DOMAIN):
        raise ValueError(f"E-mail deve pertencer ao domínio {ALLOWED_EMAIL_DOMAIN}: {email!r}")
    return normalized


def cmd_create_user(_args: argparse.Namespace) -> None:
    email = _validate_email(input("E-mail (@energisa.com.br): ").strip())
    senha = getpass.getpass("Senha: ")
    confirmacao = getpass.getpass("Confirme a senha: ")
    if senha != confirmacao:
        print("Senhas não conferem.", file=sys.stderr)
        raise SystemExit(1)

    runtime = load_firestore_runtime()
    user = create_user(runtime.client, email, hash_password(senha))
    print(f"Usuário criado: {user.email}")


def cmd_load_initial(_args: argparse.Namespace) -> None:
    if not USUARIOS_JSON_PATH.exists():
        print(
            f"Arquivo não encontrado: {USUARIOS_JSON_PATH}\n"
            "Crie-o com uma lista [{\"email\": ..., \"senha\": ...}] (ver docstring deste script).",
            file=sys.stderr,
        )
        raise SystemExit(1)

    entries = json.loads(USUARIOS_JSON_PATH.read_text(encoding="utf-8"))
    runtime = load_firestore_runtime()
    for entry in entries:
        email = _validate_email(entry["email"])
        user = create_user(runtime.client, email, hash_password(entry["senha"]))
        print(f"Usuário criado/atualizado: {user.email}")
    print(f"Carga inicial concluída ({len(entries)} usuário(s)).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gerenciamento administrativo de usuários (Task 010).")
    subparsers = parser.add_subparsers(dest="comando", required=True)

    subparsers.add_parser("create-user", help="Cria um usuário individualmente (interativo).").set_defaults(
        func=cmd_create_user
    )
    subparsers.add_parser(
        "load-initial", help="Carga inicial em lote a partir de usuarios.json (mesmo diretório)."
    ).set_defaults(func=cmd_load_initial)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
