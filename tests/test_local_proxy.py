"""Exercise local API calls with an environment proxy that returns HTTP 503."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from threading import Thread
from urllib.parse import urlsplit

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def servers():
    running = []

    def start(status=200):
        calls = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                calls.append(self.path)
                self.respond({"status": "ok", "device": "cpu"})

            def do_POST(self):
                calls.append(self.path)
                body = self.rfile.read(int(self.headers["Content-Length"]))
                assert b"test image bytes" in body
                assert urlsplit(self.path).path == "/infer/upload"
                self.respond({"device": "cpu", "size": [322, 322], "images": [
                    {"path": "image.jpg", "dimension": 8448, "norm": 1.0},
                ]})

            def respond(self, data):
                body = json.dumps(data).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        running.append((server, thread))
        return f"http://127.0.0.1:{server.server_port}", calls

    yield start
    for server, thread in running:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def proxy_environment(url):
    env = {key: value for key, value in os.environ.items() if not key.lower().endswith("_proxy")}
    env.update(http_proxy=url, HTTP_PROXY=url, https_proxy=url, HTTPS_PROXY=url)
    return env


@pytest.mark.parametrize("status,success", [(200, True), (503, False)])
def test_healthcheck_bypasses_proxy_and_checks_status(servers, status, success):
    api, api_calls = servers(status)
    proxy, proxy_calls = servers(503)
    line = next(line.strip() for line in (ROOT / "Dockerfile").read_text().splitlines()
                if line.strip().startswith("CMD python -c"))
    code = shlex.split(line)[3].replace("http://127.0.0.1:8000", api)
    result = subprocess.run([sys.executable, "-c", code], env=proxy_environment(proxy),
                            capture_output=True, text=True, timeout=10)
    assert (result.returncode == 0) == success
    assert api_calls == ["/health"]
    assert proxy_calls == []


@pytest.mark.parametrize("remote", [False, True])
def test_client_local_bypasses_proxy_remote_uses_it(servers, tmp_path, remote):
    api, api_calls = servers()
    proxy, proxy_calls = servers(200 if remote else 503)
    url = "http://api.example.invalid:8001" if remote else api
    image = tmp_path / "image.jpg"
    image.write_bytes(b"test image bytes")
    output = tmp_path / "response.json"
    result = subprocess.run([
        sys.executable, str(ROOT / "test_api.py"), str(image),
        "--url", url, "--output", str(output), "--timeout", "5",
    ], env=proxy_environment(proxy), capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["images"][0]["dimension"] == 8448
    assert len(proxy_calls if remote else api_calls) == 2
    assert (api_calls if remote else proxy_calls) == []
