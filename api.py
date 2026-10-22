"""MegaLoc HTTP API. Start with: uvicorn api:app --host 127.0.0.1 --port 8000"""

from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Annotated, Literal

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator

from inference import InferenceEngine, make_report
from test_megaloc import collect_images


class Options(BaseModel):
    height: int = Field(322, ge=14, le=1400, description="Hauteur, multiple de 14.")
    width: int = Field(322, ge=14, le=1400, description="Largeur, multiple de 14.")
    batch_size: int = Field(1, ge=1, le=64)
    top_k: int = Field(5, ge=1, le=1000)
    include_descriptors: bool = Field(True, description="Inclure les vecteurs dans la réponse JSON.")

    @field_validator("height", "width")
    @classmethod
    def patch_multiple(cls, value):
        if value % 14:
            raise ValueError("La taille doit être un multiple de 14.")
        return value


class PathRequest(Options):
    images: list[str] = Field(min_length=1, description="Chemins d'images ou dossiers sur le serveur.")
    references: list[str] = Field(default_factory=list, description="Images ou dossiers de référence sur le serveur.")
    recursive: bool = False


class Match(BaseModel):
    path: str
    cosine_similarity: float = Field(description="Similarité cosinus ; plus élevée = plus similaire, pas une probabilité.")


class ImageResult(BaseModel):
    path: str
    dimension: int
    norm: float
    descriptor: list[float] | None = None
    matches: list[Match] | None = None


class InferenceResponse(BaseModel):
    device: str
    size: list[int]
    images: list[ImageResult]


@asynccontextmanager
async def lifespan(app):
    app.state.engine = InferenceEngine(
        device=os.getenv("MEGALOC_DEVICE", "auto"),
        weights=os.getenv("MEGALOC_WEIGHTS") or None,
        cache_dir=os.getenv("MEGALOC_CACHE_DIR") or None,
        offline=os.getenv("MEGALOC_OFFLINE", "0").lower() in {"1", "true", "yes"},
        threads=int(os.environ["MEGALOC_THREADS"]) if "MEGALOC_THREADS" in os.environ else None,
    )
    yield
    del app.state.engine


app = FastAPI(
    title="MegaLoc API", version="1.0.0", lifespan=lifespan,
    description="Extraction de descripteurs et recherche d'images similaires. "
                "Le modèle ne renvoie pas directement de coordonnées GPS. "
                "Prétraitement : orientation EXIF, RGB, normalisation ImageNet et redimensionnement.",
)


@app.get("/health", tags=["État"])
def health():
    """Disponible une fois les poids chargés au démarrage."""
    return {"status": "ok", "device": str(app.state.engine.device)}


def infer(sources, names, reference_sources, reference_names, options):
    engine = app.state.engine
    size = (options.height, options.width)
    try:
        descriptors = engine.extract(sources, size, options.batch_size)
        reference_descriptors = engine.extract(reference_sources, size, options.batch_size) if reference_sources else None
        return make_report(names, descriptors, reference_names, reference_descriptors,
                           device=engine.device, size=size, top_k=options.top_k,
                           include_descriptors=options.include_descriptors)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except torch.OutOfMemoryError as error:
        raise HTTPException(503, "Mémoire insuffisante : réduire batch_size ou la taille des images.") from error


@app.post("/infer/paths", response_model=InferenceResponse, response_model_exclude_none=True,
          tags=["Inférence"], summary="Analyser des images présentes sur le serveur")
def infer_paths(request: PathRequest):
    """Les dossiers sont filtrés par extension. Le classement porte sur les références fournies."""
    try:
        images = collect_images([Path(p) for p in request.images], request.recursive)
        references = collect_images([Path(p) for p in request.references], request.recursive) if request.references else []
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    return infer(images, images, references, references, request)


@app.post("/infer/upload", response_model=InferenceResponse, response_model_exclude_none=True,
          tags=["Inférence"], summary="Envoyer des images à analyser")
def infer_upload(
    images: Annotated[list[UploadFile], File(description="Une ou plusieurs images à analyser.")],
    references: Annotated[list[UploadFile] | None, File(description="Images de référence optionnelles.")] = None,
    height: Annotated[int, Form(ge=14, le=1400, description="Hauteur, multiple de 14.")] = 322,
    width: Annotated[int, Form(ge=14, le=1400, description="Largeur, multiple de 14.")] = 322,
    batch_size: Annotated[int, Form(ge=1, le=64)] = 1,
    top_k: Annotated[int, Form(ge=1, le=1000)] = 5,
    include_descriptors: Annotated[Literal["true", "false"], Form(description="Inclure les vecteurs dans la réponse JSON.")] = "true",
):
    """Multipart : fichiers et paramètres de prétraitement, directement testables dans Swagger."""
    if height % 14 or width % 14:
        raise HTTPException(422, "height et width doivent être des multiples de 14.")
    options = Options(height=height, width=width, batch_size=batch_size, top_k=top_k,
                      include_descriptors=include_descriptors == "true")
    references = references or []
    return infer([image.file for image in images], [image.filename or "image" for image in images],
                 [image.file for image in references], [image.filename or "reference" for image in references], options)
