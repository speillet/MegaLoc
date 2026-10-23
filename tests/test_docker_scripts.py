"""Check Docker command construction without requiring a Docker daemon."""

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROXY_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "FTP_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "ftp_proxy",
    "all_proxy",
    "no_proxy",
)


@pytest.fixture
def docker_command(tmp_path):
    log = tmp_path / "command.json"
    docker = tmp_path / "docker"
    docker.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['DOCKER_TEST_COMMAND'], 'w') as output:\n"
        "    json.dump(sys.argv[1:], output)\n"
    )
    docker.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if key not in PROXY_KEYS}
    env.update(
        PATH=str(tmp_path) + os.pathsep + env["PATH"], DOCKER_TEST_COMMAND=str(log)
    )
    env.update(
        http_proxy="http://user:dummy-password@proxy.example:3128",
        https_proxy="http://user:dummy-password@proxy.example:3128",
        NO_PROXY="localhost,127.0.0.1",
    )

    def run(script, *options):
        result = subprocess.run(
            ["bash", str(ROOT / script), *options],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
        )
        args = json.loads(log.read_text()) if log.exists() else []
        return result, args

    return run


@pytest.mark.parametrize(
    "script,flag", [("docker-build.sh", "--build-arg"), ("docker-run.sh", "--env")]
)
def test_proxy_is_forwarded_without_values_in_command(docker_command, script, flag):
    result, args = docker_command(script)
    assert result.returncode == 0, result.stderr
    for key in ("http_proxy", "https_proxy", "NO_PROXY"):
        assert any(args[i : i + 2] == [flag, key] for i in range(len(args) - 1))
    assert not any("dummy-password" in value for value in args)
    assert "dummy-password" not in result.stdout + result.stderr


@pytest.mark.parametrize("script", ["docker-build.sh", "docker-run.sh"])
def test_proxy_forwarding_can_be_disabled(docker_command, script):
    result, args = docker_command(script, "--no-proxy")
    assert result.returncode == 0, result.stderr
    assert not any(key in args for key in PROXY_KEYS)


def test_host_build_network_and_repository_context(docker_command):
    result, args = docker_command("docker-build.sh", "--network", "host")
    assert result.returncode == 0, result.stderr
    assert args[args.index("--network") + 1] == "host"
    assert args[-1] == str(ROOT)


def test_invalid_network(docker_command):
    result, args = docker_command("docker-build.sh", "--network", "invalid")
    assert result.returncode != 0
    assert args == []


@pytest.mark.parametrize(
    "options,spec", [([], "all"), (["--gpus", "device=0"], "device=0")]
)
def test_run_exposes_gpus_and_preserves_port(docker_command, options, spec):
    result, args = docker_command("docker-run.sh", "--port", "8085", *options)
    assert result.returncode == 0, result.stderr
    assert args[args.index("--gpus") + 1] == spec
    assert args[args.index("--publish") + 1] == "127.0.0.1:8085:8000"


@pytest.mark.parametrize(
    "script,options",
    [
        ("docker-build.sh", ["--image"]),
        ("docker-build.sh", ["--image", "--no-cache"]),
        ("docker-run.sh", ["--port"]),
        ("docker-run.sh", ["--port", "--detach"]),
        ("docker-run.sh", ["--gpus", "--detach"]),
    ],
)
def test_missing_option_value_does_not_invoke_docker(docker_command, script, options):
    result, args = docker_command(script, *options)
    assert result.returncode != 0
    assert "Valeur manquante" in result.stderr
    assert args == []
