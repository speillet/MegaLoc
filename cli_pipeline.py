"""Extract MegaLoc descriptors and optionally rank reference images."""

import argparse
import json
from pathlib import Path


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("La valeur doit être supérieure à zéro.")
    return value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "images", nargs="+", type=Path, help="Images ou dossiers à analyser."
    )
    parser.add_argument(
        "--references",
        nargs="+",
        type=Path,
        help="Images ou dossiers de référence à classer pour chaque image.",
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Parcourir aussi les sous-dossiers."
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto (CUDA, MPS puis CPU), cpu, cuda, cuda:0 ou mps.",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=1,
        help="Images par lot (défaut : 1 ; augmenter selon la mémoire disponible).",
    )
    parser.add_argument(
        "--size",
        nargs=2,
        type=positive_int,
        default=[322, 322],
        metavar=("HEIGHT", "WIDTH"),
        help="Taille d'entrée, multiples de 14 (défaut : 322 322).",
    )
    parser.add_argument(
        "--top-k",
        type=positive_int,
        default=5,
        help="Nombre de références affichées par image (défaut : 5).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        help="Poids locaux .safetensors ; sinon téléchargement depuis gberton/MegaLoc.",
    )
    parser.add_argument("--cache-dir", type=Path, help="Dossier de cache Hugging Face.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Utiliser uniquement des poids locaux ou déjà en cache.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Enregistrer chemins, descripteurs CPU et prétraitement dans un fichier .pt.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Enregistrer les normes et les classements dans un rapport JSON.",
    )
    parser.add_argument(
        "--threads", type=positive_int, help="Nombre de threads CPU PyTorch."
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    import torch

    from inference import (
        InferenceEngine,
        collect_images,
        make_report,
        validate_image_size,
    )

    try:
        validate_image_size(args.size)
    except ValueError as error:
        parser.error(str(error))

    try:
        queries = collect_images(args.images, args.recursive)
        references = (
            collect_images(args.references, args.recursive) if args.references else []
        )
        engine = InferenceEngine(
            device=args.device,
            weights=args.weights,
            cache_dir=args.cache_dir,
            offline=args.offline,
            threads=args.threads,
        )
        device = engine.device
        print(f"Modèle chargé sur {device}", flush=True)
        descriptors = engine.extract(queries, args.size, args.batch_size)
        reference_descriptors = (
            engine.extract(references, args.size, args.batch_size)
            if references
            else None
        )
        report = make_report(
            queries,
            descriptors,
            references,
            reference_descriptors,
            device=device,
            size=args.size,
            top_k=args.top_k,
            include_descriptors=False,
        )
        for entry in report["images"]:
            print(
                f"{entry['path']} : dimension={entry['dimension']}, norme={entry['norm']:.6f}"
            )
            for rank, match in enumerate(entry.get("matches", []), 1):
                print(f"  {rank}. {match['cosine_similarity']:.4f}  {match['path']}")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "paths": queries,
                    "descriptors": descriptors,
                    "reference_paths": references,
                    "reference_descriptors": reference_descriptors,
                    "size": args.size,
                    "normalization": "ImageNet",
                },
                args.output,
            )
            print(f"Descripteurs : {args.output}")
        if args.json_output:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"Rapport : {args.json_output}")
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"Erreur : {error}\n")


if __name__ == "__main__":
    main()
