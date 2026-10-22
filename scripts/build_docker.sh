#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: build_docker.sh [options]
Construit l'image depuis la racine du dépôt, quel que soit le dossier courant.

  --image NAME          Nom/tag de l'image (défaut : megaloc-api:latest)
  --no-cache            Reconstruire sans cache Docker
  --pull                Actualiser l'image Python de base
  --network MODE        Réseau de build : default, host ou none
  --no-proxy            Ne pas transmettre les variables de proxy du terminal
  --torch-index-url URL Index des paquets PyTorch (défaut : index CPU du Dockerfile)
  -h, --help            Afficher cette aide
EOF
}

die() { echo "Erreur : $*" >&2; exit 1; }
need_value() { [[ $# -ge 2 && -n "$2" ]] || die "Valeur manquante pour $1"; }

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
image="megaloc-api:latest"
options=()
forward_proxy=true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) need_value "$@"; image="$2"; shift 2 ;;
    --no-cache|--pull) options+=("$1"); shift ;;
    --network)
      need_value "$@"
      case "$2" in default|host|none) ;; *) die "--network attend default, host ou none." ;; esac
      options+=(--network "$2"); shift 2 ;;
    --no-proxy) forward_proxy=false; shift ;;
    --torch-index-url) need_value "$@"; options+=(--build-arg "TORCH_INDEX_URL=$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Option inconnue : $1 (voir --help)" ;;
  esac
done
command -v docker >/dev/null || die "Docker est introuvable."
if "$forward_proxy"; then
  # Docker reads each value from the environment. Do not print proxy credentials
  # or bake them into the Dockerfile's ARG/ENV instructions.
  for proxy_var in HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy ftp_proxy all_proxy no_proxy; do
    if [[ -n "${!proxy_var:-}" ]]; then
      options+=(--build-arg "$proxy_var")
    fi
  done
fi
echo "Construction de $image…"
docker build "${options[@]}" --tag "$image" --file "$repo_dir/Dockerfile" "$repo_dir"
echo "Image construite : $image"
