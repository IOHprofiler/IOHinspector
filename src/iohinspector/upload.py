"""Upload a dataset to an IOHanalyzer website with a personal upload token.

The original files go up (a folder is zipped first); the server converts and
verifies them itself and keeps the original. A local convert and verify runs
first, so that a failing upload is caught before any byte is sent. The upload
id and the file's size are kept in ``<archive>.upload.json``, so that a second
run continues where the first stopped.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .convert import convert
from .verify import verify

DEFAULT_URL = "https://coaxium.liacs.nl"
_API = "/api/internal/uploads"
_WORKING = ("queued", "processing", "storing")


class UploadError(RuntimeError):
    """The upload cannot go on; the message says why. ``code`` is the HTTP status
    when the server refused."""

    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


def _log(message: str) -> None:
    print(message, file=sys.stderr)


class _Client:
    def __init__(self, url: str, token: str) -> None:
        if urllib.parse.urlsplit(url).scheme not in ("http", "https"):
            raise UploadError(f"not an http(s) URL: {url}")
        self.base = url.rstrip("/") + _API
        self.token = token

    def call(  # pylint: disable=too-many-arguments
        self,
        method: str,
        path: str = "",
        *,
        body: bytes | None = None,
        json_body: Mapping[str, Any] | None = None,
        query: str = "",
    ) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.token}"}
        if json_body is not None:
            body = json.dumps(json_body).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base}{path}{query}", body, headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:  # nosec B310
                return json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as exc:
            raise _refusal(exc) from None
        except OSError as exc:
            raise UploadError(f"cannot reach {self.base}: {exc}") from None


def _refusal(exc: urllib.error.HTTPError) -> UploadError:
    try:
        detail = json.loads(exc.read()).get("detail", "")
    except (ValueError, AttributeError):
        detail = ""
    hint = {
        401: "the token is wrong or revoked (see the website's account page)",
        413: "too large for your quota or tier",
        429: "too many uploads today; try again tomorrow",
        507: "the server's disk is full; try again later",
    }.get(exc.code, "")
    return UploadError(f"HTTP {exc.code}: {detail or hint or exc.reason}", exc.code)


def _archive(path: Path) -> Path:
    """The file to send: ``path`` itself, or for a folder ``<path>.zip``."""
    if not path.is_dir():
        return path
    target = path.with_name(path.name + ".zip")
    if not target.exists():
        _log(f"zipping {path} to {target}")
        shutil.make_archive(str(target.with_suffix("")), "zip", root_dir=path)
    return target


def check_locally(source: Path, **options: Any) -> None:
    """Convert and verify ``source``; raises ``UploadError`` when verify fails."""
    with tempfile.TemporaryDirectory() as tmp:
        dataset = Path(tmp) / "dataset"
        convert(source, dataset, **options)
        report = verify(source, dataset)
    if not report.ok:
        raise UploadError(f"verify failed, nothing uploaded:\n{report.summary()}")


def _start(client: _Client, state_file: Path, archive: Path, options: dict) -> dict:
    if state_file.exists():
        saved = json.loads(state_file.read_text())
        if saved["size"] == archive.stat().st_size:
            try:
                return client.call("GET", "/" + saved["id"])
            except UploadError as exc:
                if getattr(exc, "code", 0) != 404:
                    raise
    info = client.call(
        "POST",
        json_body={
            "filename": archive.name,
            "size": archive.stat().st_size,
            **options,
        },
    )
    state_file.write_text(json.dumps({"id": info["id"], "size": info["size"]}))
    return info


def _send(client: _Client, archive: Path, info: dict) -> dict:
    size, chunk = info["size"], info.get("chunk", 8 * 2**20)
    with archive.open("rb") as handle:
        while info["state"] == "receiving":
            offset = info["received"]
            handle.seek(offset)
            try:
                info = client.call(
                    "PUT",
                    f"/{info['id']}",
                    body=handle.read(chunk),
                    query=f"?offset={offset}",
                )
            except UploadError as exc:
                if getattr(exc, "code", 0) != 409:
                    raise
                info = client.call("GET", f"/{info['id']}")  # continue from received
            _log(f"sent {info['received']} of {size} bytes")
    return info


def _wait(client: _Client, info: dict, *, poll: float) -> dict:
    while info["state"] in _WORKING:
        time.sleep(poll)
        info = client.call("GET", f"/{info['id']}")
    return info


def upload(  # pylint: disable=too-many-arguments,too-many-locals
    path: str | Path,
    *,
    token: str,
    url: str = DEFAULT_URL,
    title: str | None = None,
    visibility: str = "private",
    tags: Sequence[str] = (),
    renames: Mapping[str, str] | None = None,
    compare_by: Sequence[str] = ("algorithm_name",),
    format: str | None = None,  # pylint: disable=redefined-builtin
    mapping: Mapping[str, Any] | None = None,
    check: bool = True,
    poll: float = 2.0,
) -> str:
    """Upload ``path`` (a file, folder or archive) and confirm it; returns the new
    dataset's id. Run again after an interruption to continue."""
    # #lizard forgives  (the command's options, passed on as they are)
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    archive = _archive(source)
    state_file = archive.with_name(archive.name + ".upload.json")
    if check and not state_file.exists():
        _log("converting and verifying locally")
        check_locally(archive, format=format, mapping=mapping)
    client = _Client(url, token)
    options = {k: v for k, v in {"format": format, "mapping": mapping}.items() if v}
    info = _start(client, state_file, archive, options)
    info = _wait(client, _send(client, archive, info), poll=poll)
    if info["state"] == "failed":
        raise UploadError(f"the server could not convert the upload: {info['error']}")
    if info["state"] == "ready":
        _log("confirming")
        info = client.call(
            "POST",
            f"/{info['id']}/confirm",
            json_body={
                "title": title or source.stem,
                "visibility": visibility,
                "tags": list(tags),
                "renames": dict(renames or {}),
                "compare_by": list(compare_by),
            },
        )
        info = _wait(client, info, poll=poll)
    if info["state"] != "done":
        raise UploadError(f"the upload is {info['state']}: {info['error']}")
    state_file.unlink(missing_ok=True)
    return info["dataset_id"]
