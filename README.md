# MegaLoc
An image retrieval model for any localization task, which achieves SOTA on most VPR datasets, including indoor and outdoor ones.

[Gradio Demo](https://gmberton.github.io/demos-url/megaloc/) - [ArXiv](https://arxiv.org/abs/2502.17237) - [Paper on ArXiv](https://arxiv.org/pdf/2502.17237) - [Paper on HF](https://huggingface.co/papers/2502.17237) - [Model on HF](https://huggingface.co/gberton/MegaLoc).

### Demo
Try the demo on your own images to see how good MegaLoc is! The demo uses a database of ~5M street-view images from San Francisco, and when you upload one it will find the most similar one from the same place.

<img width="746" height="576" alt="image" src="https://github.com/user-attachments/assets/4e7a3eec-dfee-4aae-83cc-f5146a1b421d" />


### Using the model
You can use the model with torch.hub, as simple as this
```
import torch
model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
```

Here is a complete example with the same preprocessing that we use for evaluation (ImageNet normalization, resize to 322x322)

```python
import torchvision.transforms as tfm
from PIL import Image

transform = tfm.Compose([
    tfm.ToTensor(),
    tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    tfm.Resize(size=[322, 322], antialias=True),
])

images = torch.stack([transform(Image.open(path).convert("RGB")) for path in ["im1.jpg", "im2.jpg"]])
with torch.inference_mode():
    descriptors = model(images)  # shape [2, 8448], L2-normalized

similarities = descriptors @ descriptors.T  # cosine similarities
```

For more complex uses, like computing results on VPR datasets, visualizing predictions and so on, you can use our [VPR-methods-evaluation](https://github.com/gmberton/VPR-methods-evaluation), which lets you do all this for MegaLoc and multiple other VPR methods on labelled or unlabelled datasets.

### Inference, CLI and API

The entry points follow the same layout as the PLONK project:

| File | Role |
| --- | --- |
| `inference.py` | Load the model, discover images, preprocess inputs, extract descriptors and rank references |
| `cli_pipeline.py` | Run inference locally from the command line |
| `api.py` | FastAPI endpoints and Swagger documentation |
| `Dockerfile` | Build the API runtime image |
| `docker-build.sh` | Build the Docker image |
| `docker-run.sh` | Run the API container |
| `test_api.py` | Send images to a running API |

`megaloc_model.py` contains the required model architecture. `hubconf.py` preserves
PyTorch Hub support. Runtime dependencies are in `requirements.txt`; automated
regression tests live in `tests/`.

### Installation and local inference

Requires Python 3.10 or newer:

```bash
uv venv --python 3.12 --managed-python .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate

python cli_pipeline.py image.jpg --output outputs/descriptor.pt
python cli_pipeline.py image.jpg --references ./database --recursive --top-k 5
python cli_pipeline.py --help
```

The environment is managed by `uv` in the repository's `.venv/` directory, which
is excluded from Git and the Docker build context. In fish, activate it with
`source .venv/bin/activate.fish`. Run `deactivate` to leave the environment.
If `uv` is not installed, follow the
[official installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

Inference applies EXIF orientation, RGB conversion, ImageNet normalization and a
322×322 resize. `--size HEIGHT WIDTH` accepts multiples of 14; the input must
contain more than 64 patches of 14×14 pixels for the pretrained aggregator
(112×112 is too small; 126×112 is valid). The CLI and API validate this before
inference. Other options include
`--device`, `--batch-size`, `--threads`, `--weights`, `--cache-dir`, `--offline`, and
`--json-output`. Weights are downloaded from Hugging Face on first use.
Descriptors have 8448 dimensions and L2 norm 1. Similarity scores are cosine
similarities, not probabilities; MegaLoc does not directly predict GPS coordinates.

### Docker API

The container runs on **NVIDIA GPU by default**. The image installs CUDA 13.0
PyTorch wheels from the [official CUDA index](https://pytorch.org/get-started/previous-versions/),
and sets `MEGALOC_DEVICE=cuda`. The launcher exposes GPUs using `--gpus all`.
The NVIDIA driver and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
must be installed and configured on the host. The toolkit is installed on the
host, not in the image. The CUDA wheel dependencies provide the CUDA runtime in
the container; the NVIDIA toolkit exposes the host GPU and driver libraries.

On Ubuntu/Debian, install and configure the toolkit once on the host using the
[NVIDIA instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends ca-certificates curl gnupg
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

The last command restarts the Docker daemon; run this setup before starting the
new API container. Preserve proxy variables for `sudo apt-get` if your network
requires them. This host setup requires administrator privileges.

From the repository root:

```bash
./docker-build.sh
./docker-run.sh --detach

docker logs -f megaloc-api
docker stop megaloc-api
```

Swagger: **http://127.0.0.1:8001/docs** after model loading. The default host port
is 8001, forwarded to Uvicorn on port 8000 inside the container. To select another
port and mount server-side images:

```bash
./docker-run.sh --detach --port 8085 \
  --data-dir /home/SPeillet/projects/sevilla/data
```

The data directory is mounted read-only at `/data`. The named volume
`megaloc-hf-cache` keeps downloaded weights between runs. The container runs as
UID 10001, and mounted image files must be readable by that user. Stop an existing
container named `megaloc-api` before launching another one. Use `--help` on each
Docker script for image, cache, CPU thread, offline and other options.
Use `--gpus device=0` to expose a specific host GPU. The default CUDA device is the
first GPU visible inside the container. If CUDA is inaccessible, startup fails
before loading weights rather than falling back to CPU. The build checks that
the installed PyTorch includes CUDA; a GPU is needed when running, not building.

To replace a previously built CPU container while keeping the weights cache:

```bash
./docker-build.sh
docker rm -f megaloc-api
./docker-run.sh --detach --port 8085
docker logs -f megaloc-api
```

Verify GPU access and the API after startup:

```bash
docker exec megaloc-api nvidia-smi
docker exec megaloc-api python -c 'import torch; print(torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
curl --noproxy '*' http://127.0.0.1:8085/health
```

The health response should contain `"device": "cuda"`. If Docker reports that it
cannot select a driver with GPU capabilities, follow the NVIDIA Container Toolkit
installation/configuration guide linked above. Run the Docker commands with
`sudo` if your account requires it.

The build and run scripts forward exported proxy variables. The Dockerfile
normalizes uppercase/lowercase names for APT and uv without storing proxy values
in the image. The health check bypasses proxies when calling the local API.
When using `sudo`, preserve the proxy environment according to your system policy.
The Dockerfile copies the pinned `uv` binary from its official image and installs
dependencies with `uv pip install --system`, using a BuildKit cache for downloads.
Python and PyTorch remain provided by the Python base image and the selected
PyTorch package index. See the [uv Docker guide](https://docs.astral.sh/uv/guides/integration/docker/).

### Test the API with an image

With the container running:

```bash
python test_api.py /path/to/image.jpg \
  --include-descriptors --output outputs/response.json

# If the container was launched with --port 8085:
python test_api.py /path/to/image.jpg --url http://127.0.0.1:8085
```

The client defaults to `http://127.0.0.1:8001`. It checks `/health`, uploads the
image, prints dimensions and norms, and optionally saves the JSON response.
`--references` supplies reference images, `--top-k` controls ranking,
`--mode paths` sends server-side paths instead of uploading files, and `--recursive`
scans server directories in that mode. See `python test_api.py --help`.
Local API calls bypass proxies; remote API URLs use the environment proxy settings.

### API endpoints and configuration

| Endpoint | Input | Result |
| --- | --- | --- |
| `GET /health` | None | Status and device after model loading |
| `POST /infer/upload` | Multipart images and optional references | Descriptors and reference ranking |
| `POST /infer/paths` | JSON image/reference paths or directories on the server | Descriptors and reference ranking |

Both inference endpoints accept `height`/`width` (default 322, multiples of 14,
maximum 1400), `batch_size` (1–64), `top_k` (1–1000), and `include_descriptors`.
The API loads the model once at startup and serializes model execution. Use one
Uvicorn worker. Swagger, ReDoc and the OpenAPI schema are at `/docs`, `/redoc`,
and `/openapi.json`. For a server without Docker:

```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8001
```

Server settings use environment variables: `MEGALOC_DEVICE` (`cuda` in Docker,
`auto` outside Docker),
`MEGALOC_WEIGHTS` (local checkpoint), `MEGALOC_CACHE_DIR`, `MEGALOC_OFFLINE`, and
`MEGALOC_THREADS`. Pass them using `docker-run.sh --env-file FILE` or dedicated
script options. This local API has no authentication; `/infer/paths` reads files
accessible to the server process. Invalid images/options return HTTP 422;
insufficient model memory returns HTTP 503.

### Automated checks

Tests use a small replacement model and require no weight download:

```bash
uv pip install --python .venv/bin/python -r requirements-dev.txt
ruff check --select E4,E7,E9,F,I inference.py cli_pipeline.py api.py test_api.py tests
ruff format --check inference.py cli_pipeline.py api.py test_api.py tests
python -m pytest -q
```

### FAQ

**How do I reproduce the paper's results?** With [VPR-methods-evaluation](https://github.com/gmberton/VPR-methods-evaluation) and the datasets from [VPR-datasets-downloader](https://github.com/gmberton/VPR-datasets-downloader), which has the exact splits we used ([#13](https://github.com/gmberton/MegaLoc/issues/13)).

**Will you release the training code or datasets?** No: the datasets come from sources that we are not allowed to redistribute, and we are not planning to release the training code ([#5](https://github.com/gmberton/MegaLoc/issues/5), [#9](https://github.com/gmberton/MegaLoc/issues/9)).

**Should I fine-tune MegaLoc on my data?** Usually not, unless your data is very out-of-distribution, like caves or underwater ([#6](https://github.com/gmberton/MegaLoc/issues/6)).

**How should I handle 360° images?** Crop each panorama into perspective crops and treat them as independent images ([#10](https://github.com/gmberton/MegaLoc/issues/10)).

### Qualitative examples
Here are some examples of top-1 retrieved images from the SF-XL test set, which has 2.8M images as database.

![teaser](https://github.com/user-attachments/assets/a90b8d4c-ab53-4151-aacc-93493d583713)



## Acknowledgements / Cite / BibTex

If you use this repository please cite the following
```bibtex
@inproceedings{Berton_2025_MegaLoc,
    author    = {Berton, Gabriele and Masone, Carlo},
    title     = {MegaLoc: One Retrieval to Place Them All},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
    month     = {June},
    year      = {2025},
    pages     = {2886--2892}
}
```

