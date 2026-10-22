"""Shared preprocessing and inference for the CLI and HTTP API."""

from pathlib import Path
from threading import Lock

import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file
from torchvision import transforms

from megaloc_model import MegaLoc


class InferenceEngine:
    def __init__(self, device="auto", weights=None, cache_dir=None, offline=False, threads=None):
        if threads is not None:
            if threads < 1:
                raise ValueError("threads doit être positif")
            torch.set_num_threads(threads)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        if weights is None:
            from huggingface_hub import hf_hub_download
            weights = hf_hub_download(
                repo_id="gberton/MegaLoc", filename="model.safetensors",
                cache_dir=str(cache_dir) if cache_dir else None, local_files_only=offline,
            )
        self.model = MegaLoc()
        self.model.load_state_dict(load_file(str(weights)))
        self.model = self.model.to(self.device).eval()
        self.lock = Lock()

    def extract(self, sources, size=(322, 322), batch_size=1):
        """Sources are paths or seekable binary streams; returns CPU descriptors."""
        if batch_size < 1 or len(size) != 2 or any(s < 14 or s % 14 for s in size):
            raise ValueError("Taille : deux multiples positifs de 14 ; batch_size : entier positif.")
        if not sources:
            raise ValueError("Au moins une image est requise.")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize(list(size), antialias=True),
        ])
        batches = []
        # Serialize model use to avoid concurrent GPU allocations across requests.
        with self.lock, torch.inference_mode():
            for start in range(0, len(sources), batch_size):
                tensors = []
                for source in sources[start:start + batch_size]:
                    if not isinstance(source, (str, Path)):
                        source.seek(0)
                    try:
                        with Image.open(source) as image:
                            tensors.append(transform(ImageOps.exif_transpose(image).convert("RGB")))
                    except (OSError, ValueError, Image.DecompressionBombError) as error:
                        raise ValueError(f"Image illisible : {source}") from error
                batches.append(self.model(torch.stack(tensors).to(self.device)).cpu())
        return torch.cat(batches)


def make_report(paths, descriptors, reference_paths, reference_descriptors, *, device, size,
                top_k=5, include_descriptors=True):
    report = {"device": str(device), "size": list(size), "images": []}
    for index, path in enumerate(paths):
        entry = {"path": str(path), "dimension": descriptors.shape[1],
                 "norm": descriptors[index].norm().item()}
        if include_descriptors:
            entry["descriptor"] = descriptors[index].tolist()
        if reference_descriptors is not None:
            values, indices = (reference_descriptors @ descriptors[index]).topk(min(top_k, len(reference_paths)))
            entry["matches"] = [{"path": str(reference_paths[i]), "cosine_similarity": score}
                                for score, i in zip(values.tolist(), indices.tolist())]
        report["images"].append(entry)
    return report
