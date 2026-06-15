"""Tests for hoct._models: registry, on-demand download, and inference."""

import functools
import hashlib
import http.server
import os
import socketserver
import threading

import pytest
import torch
import tracksdata as td
from typer.testing import CliRunner

from hoct import _models
from hoct._tests.conftest import GEFF_3D, MODEL_PATH, requires_geff_data, requires_model


@pytest.fixture
def served_dir(tmp_path):
    """Serve a fresh directory over localhost HTTP; yield ``(dir, base_url)``."""
    directory = tmp_path / "serve"
    directory.mkdir()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield directory, f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


def _register_blob(monkeypatch, served_dir, name, blob, *, sha256=None, cache=None):
    """Write ``blob`` to the served dir and register a model entry pointing at it."""
    directory, base_url = served_dir
    (directory / f"{name}.pt").write_bytes(blob)
    if cache is not None:
        monkeypatch.setenv("HOCT_CACHE_DIR", str(cache))
    digest = sha256 if sha256 is not None else hashlib.sha256(blob).hexdigest()
    monkeypatch.setitem(_models.MODELS, name, {"url": f"{base_url}/{name}.pt", "sha256": digest})


class TestResolveModel:
    """Resolution of model specs without any download."""

    def test_existing_path_passthrough(self, tmp_path):
        checkpoint = tmp_path / "local.pt"
        checkpoint.write_bytes(b"weights")
        assert _models.resolve_model(checkpoint) == checkpoint

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            _models.resolve_model("definitely_not_a_model")

    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _models.resolve_model(tmp_path / "missing.pt")

    def test_available_models_includes_default(self):
        assert _models.DEFAULT_MODEL in _models.available_models()


class TestModelDownload:
    """Download, hash verification, and caching via a local HTTP server."""

    def test_downloads_caches_and_reuses(self, tmp_path, served_dir, monkeypatch):
        cache = tmp_path / "cache"
        blob = b"hoct-weights-" + os.urandom(512)
        _register_blob(monkeypatch, served_dir, "dummy", blob, cache=cache)

        path = _models.resolve_model("dummy")
        assert path.exists()
        assert path.read_bytes() == blob
        assert cache in path.parents

        # Second call is served from the cache and points at the same file.
        assert _models.resolve_model("dummy") == path

    def test_bad_hash_is_rejected(self, tmp_path, served_dir, monkeypatch):
        _register_blob(monkeypatch, served_dir, "corrupt", b"payload", sha256="0" * 64, cache=tmp_path / "cache")
        with pytest.raises(ValueError, match="hash"):
            _models.resolve_model("corrupt")

    def test_load_model_downloads_and_loads_scripted_module(self, tmp_path, served_dir, monkeypatch):
        directory, base_url = served_dir
        scripted = torch.jit.script(torch.nn.Linear(3, 2))
        scripted.save(str(directory / "tiny.pt"))
        blob = (directory / "tiny.pt").read_bytes()
        monkeypatch.setenv("HOCT_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.setitem(
            _models.MODELS,
            "tiny",
            {"url": f"{base_url}/tiny.pt", "sha256": hashlib.sha256(blob).hexdigest()},
        )

        model = _models.load_model("tiny", device="cpu")
        assert isinstance(model, torch.jit.ScriptModule)
        assert model(torch.zeros(1, 3)).shape == (1, 2)


@requires_model
@requires_geff_data
def test_predict_cli_runs_inference_with_real_model(tmp_path):
    """The default pre-trained model runs end-to-end on a GEFF fixture."""
    from hoct.cli import app

    output = tmp_path / "tracks.geff"
    # `predict` takes the GEFF and the model as positional arguments.
    result = CliRunner().invoke(
        app,
        ["predict", GEFF_3D, str(MODEL_PATH), "-o", str(output), "--device", "cpu"],
    )

    assert result.exit_code == 0, f"predict failed:\n{result.stdout}"
    graph, _ = td.graph.InMemoryGraph.from_geff(str(output))
    assert graph.num_nodes() > 0
    assert graph.num_edges() > 0
    # The model populated edge predictions on the saved (full) candidate graph.
    assert "solution" in graph.edge_attr_keys()
    assert "similarity" in graph.edge_attr_keys()
