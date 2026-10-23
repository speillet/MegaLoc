"""HTTP integration tests with real preprocessing and a small replacement model."""

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image
import pytest
from safetensors.torch import save_file
import torch

import api
import inference


def image_bytes(color):
    buffer = BytesIO()
    Image.new("RGB", (60, 80), color).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def client(monkeypatch, tmp_path):
    class SmallModel(torch.nn.Module):
        def forward(self, images):
            return torch.nn.functional.normalize(images.mean(dim=(2, 3)), dim=1)

    monkeypatch.setattr(inference, "MegaLoc", SmallModel)
    weights = tmp_path / "weights.safetensors"
    save_file({}, weights)
    monkeypatch.setenv("MEGALOC_WEIGHTS", str(weights))
    monkeypatch.setenv("MEGALOC_DEVICE", "cpu")
    with TestClient(api.app) as client:
        yield client


def test_upload_ranking_and_descriptor(client):
    response = client.post("/infer/upload", files=[
        ("images", ("query.png", image_bytes("red"), "image/png")),
        ("references", ("other.png", image_bytes("blue"), "image/png")),
        ("references", ("same.png", image_bytes("red"), "image/png")),
    ], data={"batch_size": "2", "top_k": "1"})
    assert response.status_code == 200, response.text
    result = response.json()["images"][0]
    assert result["norm"] == pytest.approx(1)
    assert len(result["descriptor"]) == result["dimension"] == 3
    assert result["matches"][0]["path"] == "same.png"
    assert result["matches"][0]["cosine_similarity"] == pytest.approx(1)
    assert len(result["matches"]) == 1


def test_paths_and_optional_descriptors(client, tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(image_bytes("red"))
    response = client.post("/infer/paths", json={
        "images": [str(tmp_path)], "references": [str(image)],
        "include_descriptors": False, "top_k": 5,
    })
    assert response.status_code == 200, response.text
    result = response.json()["images"][0]
    assert "descriptor" not in result
    assert len(result["matches"]) == 1
    assert result["path"] == str(image)


@pytest.mark.parametrize("parameters", [
    {"height": 320}, {"batch_size": 0}, {"top_k": 0}, {"width": 1401},
])
def test_invalid_options(client, parameters):
    assert client.post("/infer/paths", json={"images": ["unused"], **parameters}).status_code == 422


def test_bad_inputs(client):
    assert client.post("/infer/paths", json={"images": []}).status_code == 422
    assert client.post("/infer/paths", json={"images": ["/missing/image.jpg"]}).status_code == 422
    assert client.post("/infer/upload", files={"images": ("bad.jpg", b"not an image")}).status_code == 422
    assert client.post("/infer/upload", files={"images": ("ok.png", image_bytes("red"))},
                       data={"height": 320}).status_code == 422
    assert client.post("/infer/upload").status_code == 422


def test_health_and_documentation(client):
    assert client.get("/health").json() == {"status": "ok", "device": "cpu"}
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert "multipart/form-data" in schema["paths"]["/infer/upload"]["post"]["requestBody"]["content"]
    assert "/infer/paths" in schema["paths"]


def test_out_of_memory_returns_503(client, monkeypatch):
    def fail(*args, **kwargs):
        raise torch.OutOfMemoryError("test")
    monkeypatch.setattr(api.app.state.engine, "extract", fail)
    response = client.post("/infer/upload", files={"images": ("image.png", image_bytes("red"))})
    assert response.status_code == 503


def test_cuda_startup_fails_before_loading_weights_without_gpu(monkeypatch):
    monkeypatch.setenv("MEGALOC_DEVICE", "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA demandé mais inaccessible"):
        with TestClient(api.app):
            pass


def test_invalid_cuda_index_fails_before_loading_weights(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="GPU cuda:1 indisponible"):
        inference.InferenceEngine(device="cuda:1")
