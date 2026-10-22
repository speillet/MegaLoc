"""Test persistent Docker proxy configuration without touching user settings."""

import json
import os
from pathlib import Path
import stat
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/configure_docker_proxy.py"


def run_setup(config, **proxies):
    env = {key: value for key, value in os.environ.items() if not key.lower().endswith("_proxy")}
    env.update(proxies)
    return subprocess.run([sys.executable, str(SCRIPT), "--config", str(config)],
                          env=env, capture_output=True, text=True)


def test_proxy_config_preserves_existing_docker_settings(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"credsStore": "example", "proxies": {
        "default": {"noProxy": "localhost"}, "https://other-daemon": {"httpProxy": "http://other:3128"},
    }}))
    proxy = "http://user:dummy-password@proxy.example:3128"
    result = run_setup(path, http_proxy=proxy, https_proxy=proxy)
    assert result.returncode == 0, result.stderr
    saved = json.loads(path.read_text())
    assert saved["credsStore"] == "example"
    assert saved["proxies"]["https://other-daemon"]["httpProxy"] == "http://other:3128"
    assert saved["proxies"]["default"] == {"httpProxy": proxy, "httpsProxy": proxy, "noProxy": "localhost"}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert "dummy-password" not in result.stdout + result.stderr


def test_create_config_from_uppercase_proxy(tmp_path):
    path = tmp_path / "docker" / "config.json"
    result = run_setup(path, HTTPS_PROXY="http://proxy.example:3128", NO_PROXY="localhost,127.0.0.1")
    assert result.returncode == 0, result.stderr
    assert json.loads(path.read_text())["proxies"]["default"] == {
        "httpsProxy": "http://proxy.example:3128", "noProxy": "localhost,127.0.0.1",
    }


def test_no_proxy_does_not_create_config(tmp_path):
    path = tmp_path / "config.json"
    assert run_setup(path, NO_PROXY="localhost").returncode == 1
    assert not path.exists()


def test_invalid_config_is_not_overwritten(tmp_path):
    path = tmp_path / "config.json"
    original = '{"sensitive-dummy-value":'
    path.write_text(original)
    result = run_setup(path, http_proxy="http://proxy.example:3128")
    assert result.returncode == 1
    assert path.read_text() == original
    assert "sensitive-dummy-value" not in result.stdout + result.stderr
