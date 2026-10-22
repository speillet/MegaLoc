#!/usr/bin/env python3
"""Persist the current proxy environment in the Docker client configuration."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


PROXY_FIELDS = {
    "httpProxy": ("http_proxy", "HTTP_PROXY"),
    "httpsProxy": ("https_proxy", "HTTPS_PROXY"),
    "ftpProxy": ("ftp_proxy", "FTP_PROXY"),
    "allProxy": ("all_proxy", "ALL_PROXY"),
    "noProxy": ("no_proxy", "NO_PROXY"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path(os.getenv("DOCKER_CONFIG", str(Path.home() / ".docker"))) / "config.json",
        help="Fichier du client Docker (défaut : $DOCKER_CONFIG/config.json ou ~/.docker/config.json).",
    )
    args = parser.parse_args()
    settings = {}
    for field, names in PROXY_FIELDS.items():
        value = next((os.environ[name] for name in names if os.environ.get(name)), None)
        if value is not None:
            settings[field] = value
    if not any(field in settings for field in ("httpProxy", "httpsProxy", "ftpProxy", "allProxy")):
        parser.exit(1, "Erreur : aucun proxy exporté dans ce terminal. Définir http_proxy et/ou https_proxy avant de relancer.\n")

    temporary = None
    try:
        config = json.loads(args.config.read_text(encoding="utf-8")) if args.config.exists() else {}
        if not isinstance(config, dict):
            raise ValueError("root")
        proxies = config.setdefault("proxies", {})
        if not isinstance(proxies, dict):
            raise ValueError("proxies")
        default = proxies.setdefault("default", {})
        if not isinstance(default, dict):
            raise ValueError("default")
        default.update(settings)
        args.config.parent.mkdir(parents=True, exist_ok=True)
        # Atomic replacement with mode 0600; keep other Docker settings intact.
        with tempfile.NamedTemporaryFile(mode="w", dir=args.config.parent, encoding="utf-8", delete=False) as output:
            temporary = Path(output.name)
            json.dump(config, output, indent=2)
            output.write("\n")
        os.replace(temporary, args.config)
    except (OSError, ValueError):
        # JSON error details can contain private config values: do not print them.
        print("Erreur : configuration Docker illisible, invalide ou non modifiable ; fichier conservé.", file=sys.stderr)
        return 1
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Proxy du client Docker configuré : {args.config}")
    print("Paramètres repris : " + ", ".join(sorted(settings)))
    print("Applicable aux prochains builds et conteneurs, sans redémarrage de Docker.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
