#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: build_docker.sh [options]
Construit l'image depuis la racine du dépôt, quel que soit le dossier courant.

  --image NAME          Nom/tag de l'image (défaut : megaloc-api:latest)
  --no-cache            Reconstruire sans cache Docker
  --pull                Actualiser l'image Python de base
  --torch-index-url URL Index des paquets PyTorch (défaut : index CPU du Dockerfile)
  -h, --help            Afficher cette aide
EOF
}

die() { echo "Erreur : $*" >&2; exit 1; }
need_value() { [[ $# -ge 2 && -n "$2" ]] || die "Valeur manquante pour $1"; }

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
image="megaloc-api:latest"
options=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) need_value "$@"; image="$2"; shift 2 ;;
    --no-cache|--pull) options+=("$1"); shift ;;
    --torch-index-url) need_value "$@"; options+=(--build-arg "TORCH_INDEX_URL=$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Option inconnue : $1 (voir --help)" ;;
  esac
done
command -v docker >/dev/null || die "Docker est introuvable."
echo "Construction de $image…"
docker build "${options[@]}" --tag "$image" --file "$repo_dir/Dockerfile" "$repo_dir"
echo "Image construite : $image"
