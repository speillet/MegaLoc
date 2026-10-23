"""Envoie une image à l'API MegaLoc et affiche le résultat, sans charger le modèle localement."""

import argparse
from contextlib import ExitStack
from ipaddress import ip_address
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit

import httpx


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Exemple : python test_api.py image.jpg --output resultat.json",
    )
    parser.add_argument("images", nargs="+", help="Images locales (upload) ou chemins côté serveur (paths).")
    parser.add_argument("--url", default="http://127.0.0.1:8001", help="URL de base de l'API (défaut : http://127.0.0.1:8001).")
    parser.add_argument("--mode", choices=["upload", "paths"], default="upload")
    parser.add_argument("--references", nargs="+", default=[])
    parser.add_argument("--recursive", action="store_true", help="Parcourir les dossiers côté serveur en mode paths.")
    parser.add_argument("--size", nargs=2, type=int, default=[322, 322], metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--include-descriptors", action="store_true", help="Inclure les vecteurs dans la réponse enregistrée.")
    parser.add_argument("--timeout", type=float, default=300, help="Timeout HTTP en secondes (défaut : 300).")
    parser.add_argument("--output", type=Path, help="Enregistrer la réponse JSON complète.")
    args = parser.parse_args()
    if any(s < 14 or s > 1400 or s % 14 for s in args.size):
        parser.error("--size attend deux multiples de 14 entre 14 et 1400.")
    if not 1 <= args.batch_size <= 64 or not 1 <= args.top_k <= 1000 or args.timeout <= 0:
        parser.error("batch-size : 1–64 ; top-k : 1–1000 ; timeout : positif.")
    if args.recursive and args.mode != "paths":
        parser.error("--recursive nécessite --mode paths.")
    options = {"height": args.size[0], "width": args.size[1], "batch_size": args.batch_size,
               "top_k": args.top_k, "include_descriptors": args.include_descriptors}
    try:
        host = (urlsplit(args.url).hostname or "").lower().rstrip(".")
        try:
            local_api = ip_address(host).is_loopback
        except ValueError:
            local_api = host == "localhost"
        with httpx.Client(base_url=args.url.rstrip("/") + "/", timeout=args.timeout,
                          trust_env=not local_api) as client:
            health = client.get("health")
            health.raise_for_status()
            print("État de l'API :", health.json())
            if args.mode == "paths":
                response = client.post("infer/paths", json={**options, "images": args.images,
                                       "references": args.references, "recursive": args.recursive})
            else:
                with ExitStack() as stack:
                    files = [(field, (Path(path).name, stack.enter_context(open(path, "rb")), "application/octet-stream"))
                             for field, paths in (("images", args.images), ("references", args.references))
                             for path in paths]
                    data = {key: str(value).lower() if isinstance(value, bool) else str(value)
                            for key, value in options.items()}
                    response = client.post("infer/upload", files=files, data=data)
            response.raise_for_status()
            report = response.json()
        print(f"HTTP {response.status_code} — périphérique : {report['device']}, taille : {report['size']}")
        for image in report["images"]:
            print(f"{image['path']} : dimension={image['dimension']}, norme={image['norm']:.6f}")
            for rank, match in enumerate(image.get("matches", []), 1):
                print(f"  {rank}. {match['cosine_similarity']:.4f}  {match['path']}")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            print("Réponse enregistrée :", args.output)
    except httpx.HTTPStatusError as error:
        print(f"Erreur HTTP {error.response.status_code} : {error.response.text}", file=sys.stderr)
        return 1
    except (httpx.RequestError, OSError, ValueError, KeyError) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
