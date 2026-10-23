#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: docker-run.sh [options]
Lance l'API sur GPU NVIDIA et conserve les poids dans un volume Docker nommé.

  --image NAME       Image à lancer (défaut : megaloc-api:latest)
  --name NAME        Nom du conteneur (défaut : megaloc-api)
  --port PORT        Port local (défaut : 8001)
  --bind ADDRESS     Adresse d'écoute sur l'hôte (défaut : 127.0.0.1)
  --gpus SPEC        GPU exposés à Docker : all ou device=0 (défaut : all)
  --cache-volume V   Volume des poids (défaut : megaloc-hf-cache)
  --data-dir DIR    Dossier hôte monté en lecture seule dans /data
  --threads N       Nombre de threads CPU PyTorch
  --offline         Exiger des poids déjà en cache
  --env-file FILE   Variables MEGALOC_* dans un fichier d'environnement
  --no-proxy        Ne pas transmettre les variables de proxy du terminal
  --detach          Lancer en arrière-plan
  -h, --help         Afficher cette aide

Arrêter : docker stop megaloc-api
Logs    : docker logs -f megaloc-api
EOF
}

die() { echo "Erreur : $*" >&2; exit 1; }
need_value() { [[ $# -ge 2 && -n "$2" ]] || die "Valeur manquante pour $1"; }

image="megaloc-api:latest"
name="megaloc-api"
port="8001"
bind="127.0.0.1"
gpus="all"
cache_volume="megaloc-hf-cache"
options=()
env_options=()
forward_proxy=true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) need_value "$@"; image="$2"; shift 2 ;;
    --name) need_value "$@"; name="$2"; shift 2 ;;
    --port) need_value "$@"; port="$2"; shift 2 ;;
    --bind) need_value "$@"; bind="$2"; shift 2 ;;
    --gpus) need_value "$@"; gpus="$2"; shift 2 ;;
    --cache-volume) need_value "$@"; cache_volume="$2"; shift 2 ;;
    --data-dir)
      need_value "$@"
      [[ -d "$2" ]] || die "Dossier introuvable : $2"
      data_dir="$(cd -- "$2" && pwd)"
      options+=(--mount "type=bind,source=$data_dir,target=/data,readonly")
      shift 2 ;;
    --threads)
      need_value "$@"
      [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "--threads doit être un entier positif."
      env_options+=(--env "MEGALOC_THREADS=$2")
      shift 2 ;;
    --offline) env_options+=(--env MEGALOC_OFFLINE=1); shift ;;
    --env-file)
      need_value "$@"
      [[ -f "$2" ]] || die "Fichier introuvable : $2"
      options+=(--env-file "$2"); shift 2 ;;
    --detach) options+=(--detach); shift ;;
    --no-proxy) forward_proxy=false; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Option inconnue : $1 (voir --help)" ;;
  esac
done
[[ "$port" =~ ^[0-9]{1,5}$ ]] || die "--port doit être un entier entre 1 et 65535."
(( 10#$port >= 1 && 10#$port <= 65535 )) || die "--port doit être entre 1 et 65535."
command -v docker >/dev/null || die "Docker est introuvable."
if "$forward_proxy"; then
  for proxy_var in HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy ftp_proxy all_proxy no_proxy; do
    if [[ -n "${!proxy_var:-}" ]]; then
      env_options+=(--env "$proxy_var")
    fi
  done
fi
echo "Swagger après chargement du modèle : http://$bind:$port/docs"
exec docker run --rm --name "$name" \
  --gpus "$gpus" \
  --publish "$bind:$port:8000" \
  --mount "type=volume,source=$cache_volume,target=/home/megaloc/.cache/huggingface" \
  "${options[@]}" "${env_options[@]}" "$image"
