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

### Local command line

Run from this repository with the Python environment activated:

```bash
# Extract and save one image descriptor
python test_megaloc.py image.jpg --output descriptor.pt

# Rank reference images for each query image
python test_megaloc.py image.jpg --references ./database --recursive --top-k 5 --json-output matches.json

# Process a directory in batches
python test_megaloc.py ./images --device cuda --batch-size 4 --output descriptors.pt

python test_megaloc.py --help
```

The default preprocessing matches the example above (ImageNet normalization and
322×322 resize); EXIF orientation is applied before conversion to RGB. Custom
`--size HEIGHT WIDTH` values must be multiples of 14. Device selection is automatic
(CUDA, MPS, then CPU); `--threads` controls CPU threading. Pretrained architecture
dimensions are fixed to match the weights.

Weights are downloaded from Hugging Face on first use. Use `--cache-dir` for a
custom cache, `--offline` to require cached weights, or `--weights model.safetensors`
for a local checkpoint. The `.pt` output contains image paths, CPU descriptors,
reference descriptors when supplied, and preprocessing metadata. The JSON report
contains descriptor norms and top reference matches with cosine similarities
(higher means more similar, not a probability). MegaLoc does not return GPS
coordinates directly. Matching images present in both inputs can match themselves.

### HTTP API (FastAPI + Swagger)

Requires Python 3.10 or newer. From the repository, activate your environment and run:

```bash
python -m pip install -r requirements-api.txt
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/docs** for interactive Swagger documentation: expand
an endpoint, click **Try it out**, supply inputs, then **Execute**. ReDoc is at
`/redoc` and the OpenAPI schema at `/openapi.json`. The model loads once during
startup; the first launch downloads weights if they are not cached. Use one worker
to keep one model instance in memory.

| Endpoint | Input | Output |
| --- | --- | --- |
| `GET /health` | None | Server status and device after model loading |
| `POST /infer/paths` | JSON with server-side image paths/directories | Descriptors and optional reference ranking |
| `POST /infer/upload` | Multipart image files and form parameters | Descriptors and optional reference ranking |

Upload an image from any client:

```bash
curl http://127.0.0.1:8000/infer/upload \
  -F 'images=@/path/to/image.jpg' \
  -F 'height=322' -F 'width=322' \
  -F 'include_descriptors=true'
```

Repeat `-F 'images=@...'` for multiple query images. Add one or more
`-F 'references=@...'` fields to rank reference images, with `-F 'top_k=5'`.
Uploads are processed from temporary streams, without permanent image files.

For files already on the server:

```bash
curl http://127.0.0.1:8000/infer/paths \
  -H 'Content-Type: application/json' \
  -d '{"images": ["/path/to/image.jpg"], "references": ["/path/to/database"], "recursive": true, "batch_size": 1, "top_k": 5, "include_descriptors": false}'
```

Both endpoints accept `height` and `width` (default 322, multiples of 14, maximum
1400), `batch_size` (1–64, default 1), `top_k` (1–1000, default 5), and
`include_descriptors` (default true). Responses contain `device`, `size`, and
`images`; each image has `path` (filename for uploads), `dimension`, `norm`, an
optional `descriptor`, and `matches` when references are supplied. Each match
contains `path` and `cosine_similarity`. Set `include_descriptors=false` for smaller
responses. The API returns JSON directly; save it client-side if needed.

Unlike per-request image options, device and weight settings apply to the entire
server and are configured before startup through environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MEGALOC_DEVICE` | `auto` | `cpu`, `cuda`, `cuda:0`, `mps`, or automatic selection |
| `MEGALOC_WEIGHTS` | Hugging Face download/cache | Local `.safetensors` checkpoint |
| `MEGALOC_CACHE_DIR` | Hugging Face default | Weight cache directory |
| `MEGALOC_OFFLINE` | `0` | `1` to require cached/local weights |
| `MEGALOC_THREADS` | PyTorch default | Positive number of CPU threads |

For example:

```bash
MEGALOC_DEVICE=cpu MEGALOC_THREADS=4 python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Invalid inputs or unreadable images return HTTP 422. Insufficient model memory
returns HTTP 503; reduce image size or batch size. Model execution is serialized
across concurrent requests. This local API has no authentication; `/infer/paths`
reads paths accessible to the server process.

The uploads follow the [FastAPI file/form conventions](https://fastapi.tiangolo.com/tutorial/request-files/).
To run HTTP integration tests (small replacement model; no weight download):

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

### Manual API client

With the API running, install the lightweight client dependency and send an image:

```bash
python -m pip install 'httpx>=0.27,<1'
python test_api_client.py /path/to/image.jpg --output response.json
```

Use `--url http://127.0.0.1:8000` to select a server, `--references ref1.jpg ref2.jpg`
to compare images, or `--mode paths` to send server-side paths instead of files.
`--size HEIGHT WIDTH`, `--batch-size`, `--top-k`, `--timeout`, and
`--include-descriptors` customize the request. `--recursive` is available in paths
mode. The script checks `/health`, prints a compact result, and exits with a nonzero
code on connection or HTTP errors. See `python test_api_client.py --help`.

### Docker

Build and start with the provided scripts (they also work from another directory):

```bash
./scripts/build_docker.sh
./scripts/run_docker.sh --detach

docker logs -f megaloc-api
docker stop megaloc-api
```

Swagger: **http://127.0.0.1:8000/docs** after model loading. To change the image tag,
port, or mount the test data:

```bash
./scripts/build_docker.sh --image megaloc-api:dev
./scripts/run_docker.sh --image megaloc-api:dev --port 8080 \
  --data-dir /home/SPeillet/projects/sevilla/data --threads 4 --detach
```

Use container paths starting with `/data` for `/infer/paths`. The scripts preserve
the weight cache in `megaloc-hf-cache` by default. `--offline` requires weights
already present in that cache. `--env-file` passes server configuration through a
Docker environment file. See each script's `--help` for all options. By default,
running in the foreground displays logs; press Ctrl+C to stop. An existing
container with the same name must be stopped before starting another one.

Build and run the API on CPU:

```bash
docker build -t megaloc-api .
docker run --rm --name megaloc-api \
  -p 127.0.0.1:8000:8000 \
  -v megaloc-hf-cache:/home/megaloc/.cache/huggingface \
  megaloc-api
```

Swagger is available at **http://127.0.0.1:8000/docs** once the model has loaded.
The first launch downloads the weights. The named volume preserves them across
container restarts; weights are not embedded in the image. The image installs
CPU wheels from the [official PyTorch index](https://docs.pytorch.org/get-started/previous-versions/).
The container runs as UID 10001 with one Uvicorn worker and a `/health` health check.

Test it from another terminal with a file on the host:

```bash
python test_api_client.py /path/to/image.jpg --output response.json
```

Uploads work without mounting image directories. For `/infer/paths`, mount your
data read-only and use the paths **inside the container**, such as `/data/image.jpg`:

```bash
docker run --rm --name megaloc-api \
  -p 127.0.0.1:8000:8000 \
  -v megaloc-hf-cache:/home/megaloc/.cache/huggingface \
  -v /home/SPeillet/projects/sevilla/data:/data:ro \
  -e MEGALOC_THREADS=4 \
  megaloc-api
```

The mounted files must be readable by UID 10001. Pass API settings with `-e`, for
example `-e MEGALOC_OFFLINE=1` after the weights have been cached. Local weights can
also be mounted read-only and selected with `-e MEGALOC_WEIGHTS=/weights/model.safetensors`.
The `.dockerignore` excludes the Python environment, Git history, and local outputs.
For Docker run options, see the [official reference](https://docs.docker.com/engine/containers/run/).

If a build reports `Unable to locate package libgomp1`, check the preceding APT
messages: a timeout fetching package indexes can cause this error. The Dockerfile
uses [Debian's HTTPS repositories](https://wiki.debian.org/SourcesList), retries
downloads, and stops if any package index cannot be updated.

On networks requiring a proxy, the build and run scripts automatically forward
exported proxy variables (`HTTP_PROXY`, `HTTPS_PROXY`, `FTP_PROXY`, `ALL_PROXY`,
`NO_PROXY`, and their lowercase equivalents). This lets APT/pip download packages
and lets the API download model weights at startup. The scripts pass variable
names to Docker, without printing their values. Proxy values are not added to the
Dockerfile or baked into the image. See [Docker's proxy documentation](https://docs.docker.com/engine/cli/proxy/).
Use `--no-proxy` to disable forwarding from the terminal; any proxy settings in
Docker's own configuration or `--env-file` still apply. If using `sudo`, ensure
the proxy environment is preserved according to your system's sudo policy.

If HTTPS still times out inside the build but works on the host, try host
networking on Linux:

```bash
./scripts/build_docker.sh --network host
```

This uses the Docker builder host's network for build steps only; it does not
change the API's port mapping. A remote builder uses its own host's network.
Custom BuildKit builders may require the `network.host` entitlement. A proxy
address pointing at `localhost` must also be reachable from inside the container;
in a bridged container `localhost` refers to that container. A build warning about
missing Git commit information is separate from APT connectivity errors.

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

