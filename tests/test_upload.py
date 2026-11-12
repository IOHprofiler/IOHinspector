"""Tests of ``iohinspector upload`` against a small local imitation of the server."""

from __future__ import annotations

import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from iohinspector import upload as up
from iohinspector.cli import main

TESTS = Path(__file__).resolve().parent
IOH = TESTS / "io" / "fixtures" / "ioh040"
CHUNK = 1000
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


class Fake:
    """The state of one fake server; ``fail`` maps a request to an HTTP status."""

    def __init__(self) -> None:
        self.data = b""
        self.size = 0
        self.state = "receiving"
        self.fail: dict[str, int] = {}
        self.log: list[str] = []
        self.confirmed: dict = {}
        self.started: dict = {}

    def info(self) -> dict:
        return {
            "id": "u1",
            "size": self.size,
            "received": len(self.data),
            "chunk": CHUNK,
            "state": self.state,
            "error": "",
            "dataset_id": "d1" if self.state == "done" else None,
        }


def _handler(fake: Fake):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence
            pass

        def _reply(self, code: int, body: dict | None = None):
            raw = json.dumps(body or {}).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _body(self) -> bytes:
            return self.rfile.read(int(self.headers.get("Content-Length", 0)))

        def _serve(self):
            key = f"{self.command} {self.path.split('?')[0]}"
            fake.log.append(key)
            assert self.headers["Authorization"] == "Bearer ioh_secret"
            if key in fake.fail:
                return self._reply(fake.fail[key], {"detail": "refused " + key})
            if self.command == "POST" and self.path.endswith("/uploads"):
                fake.started = json.loads(self._body())
                fake.size = fake.started["size"]
                return self._reply(200, fake.info())
            if self.command == "GET":
                return self._reply(200, fake.info())
            if self.command == "PUT":
                offset = int(self.path.split("offset=")[1])
                if offset != len(fake.data):
                    return self._reply(409, {"detail": "continues elsewhere"})
                fake.data += self._body()
                if len(fake.data) == fake.size:
                    fake.state = "ready"
                return self._reply(200, fake.info())
            fake.confirmed = json.loads(self._body())
            fake.state = "done"
            return self._reply(200, fake.info())

        do_GET = do_PUT = do_POST = _serve

    return Handler


@pytest.fixture(name="server")
def _server():
    fake = Fake()
    httpd = HTTPServer(("127.0.0.1", 0), _handler(fake))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    fake.url = f"http://127.0.0.1:{httpd.server_port}"  # type: ignore[attr-defined]
    yield fake
    httpd.shutdown()


@pytest.fixture(name="archive")
def _archive(tmp_path: Path) -> Path:
    shutil.make_archive(str(tmp_path / "data"), "zip", IOH)
    return tmp_path / "data.zip"


def _run(fake, path, **kw):
    return up.upload(path, token="ioh_secret", url=fake.url, poll=0, **kw)


def test_upload_in_chunks_and_confirm(server, archive, capsys):
    assert _run(server, archive, title="T", tags=["a=1"]) == "d1"
    assert server.data == archive.read_bytes()
    assert server.log.count("PUT /api/internal/uploads/u1") > 1
    assert server.confirmed["title"] == "T" and server.confirmed["tags"] == ["a=1"]
    assert not archive.with_name("data.zip.upload.json").exists()
    assert "sent" in capsys.readouterr().err


def test_folder_is_zipped(server, tmp_path):
    folder = tmp_path / "run"
    shutil.copytree(IOH, folder)
    assert _run(server, folder) == "d1"
    assert server.started["filename"] == "run.zip"
    assert server.data[:2] == b"PK"


def test_resume_after_interruption(server, archive):
    server.fail["PUT /api/internal/uploads/u1"] = 500
    with pytest.raises(up.UploadError, match="500"):
        _run(server, archive)
    assert server.data == b""
    server.fail.clear()
    # one chunk arrives, then the connection "breaks" before the client sees it
    server.data = archive.read_bytes()[:CHUNK]
    assert _run(server, archive) == "d1"
    assert server.data == archive.read_bytes()
    assert server.log.count("POST /api/internal/uploads") == 1


def test_409_resyncs(server, archive):
    server.size = archive.stat().st_size
    client = up._Client(server.url, "ioh_secret")
    info = {**server.info(), "received": 0}
    server.data = archive.read_bytes()[:CHUNK]  # the client believes 0 is received
    info = up._send(client, archive, info)
    assert info["state"] == "ready" and server.data == archive.read_bytes()


@pytest.mark.parametrize(
    "code, text",
    [(413, "refused"), (429, "refused"), (507, "refused")],
)
def test_quota_errors(server, archive, code, text):
    server.fail["POST /api/internal/uploads"] = code
    with pytest.raises(up.UploadError, match=f"HTTP {code}: {text}"):
        _run(server, archive)


def test_error_hint_without_detail():
    err = up._refusal(
        up.urllib.error.HTTPError("u", 429, "Too Many", {}, None)  # type: ignore
    )
    assert "tomorrow" in str(err)


def test_failed_conversion(server, archive):
    server.state = "failed"
    with pytest.raises(up.UploadError, match="could not convert"):
        _run(server, archive)


def test_refused_when_verify_fails(server, archive, monkeypatch):
    class Bad:
        ok = False

        @staticmethod
        def summary():
            return "mismatch"

    monkeypatch.setattr(up, "verify", lambda *a, **k: Bad)
    with pytest.raises(up.UploadError, match="mismatch"):
        _run(server, archive)
    assert server.log == []


def test_cli(server, archive, monkeypatch, capsys):
    monkeypatch.setenv("IOH_TOKEN", "ioh_secret")
    argv = ["upload", str(archive), "--url", server.url, "--rename", "a=b"]
    assert main(argv) == 0
    assert "dataset d1" in capsys.readouterr().out
    assert server.confirmed["renames"] == {"a": "b"}


def test_cli_errors(server, archive, monkeypatch, capsys):
    monkeypatch.delenv("IOH_TOKEN", raising=False)
    assert main(["upload", str(archive)]) == 2
    assert "no token" in capsys.readouterr().err
    server.fail["POST /api/internal/uploads"] = 507
    argv = ["upload", str(archive), "--url", server.url, "--token", "ioh_secret"]
    assert main(argv + ["--no-check"]) == 2
    assert "507" in capsys.readouterr().err
