"""Tests for scripts/datasets/download.py: resumable segmented downloads (against a local HTTP server with range
support), split archives, and the path checks of the extraction."""

import http.server
import importlib.util
import io
import os
import shutil
import subprocess
import tarfile
import threading
import zipfile

import pytest

from conftest import ROOT

spec = importlib.util.spec_from_file_location("download", os.path.join(ROOT, "scripts", "datasets", "download.py"))
dl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dl)


class RangeHandler(http.server.BaseHTTPRequestHandler):
    """Serves the bytes in self.server.payload, honouring "Range: bytes=a-" and "bytes=a-b"."""

    def log_message(self, *args):
        pass

    def do_GET(self):
        data = self.server.payload
        rng = self.headers.get("Range")
        if rng:
            a, b = rng.split("=", 1)[1].split("-")
            a, b = int(a), int(b) if b else len(data) - 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
            body = data[a:b + 1]
        else:
            self.send_response(200)
            body = data
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    srv.payload = os.urandom(3_000_001)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


def entry(srv, **kw):
    e = dict(url=f"http://127.0.0.1:{srv.server_address[1]}/f.bin", name="f.bin", size=len(srv.payload),
             algo="sha256", digest=__import__("hashlib").sha256(srv.payload).hexdigest(), into=None, group=None)
    e.update(kw)
    return e


def test_segmented_download_resumes_from_its_state_file(server, tmp_path):
    e = entry(server)
    part = str(tmp_path / "f.bin.part")
    # An interrupted run: the first segment complete, the second half done, the third untouched
    step = -(-e["size"] // 3)
    with open(part, "wb") as f:
        f.truncate(e["size"])
        f.seek(0)
        f.write(server.payload[:step])
        f.seek(step)
        f.write(server.payload[step:step + step // 2])
    segs = [[0, step, step], [step, 2 * step, step // 2], [2 * step, e["size"], 0]]
    with open(part + ".json", "w") as f:
        dl.json.dump(dict(size=e["size"], segments=segs), f)
    dl.fetch_segmented(e, part, 3, retries=2)
    assert open(part, "rb").read() == server.payload
    assert not os.path.exists(part + ".json")


def test_metadata_requests_retry_transient_errors_only(monkeypatch):
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise dl.urllib.error.URLError(OSError(-3, "Temporary failure in name resolution"))
        return "ok"
    assert dl.with_retries(flaky) == "ok" and len(calls) == 3

    def missing():
        calls.append(1)
        raise dl.urllib.error.HTTPError("http://x", 404, "Not Found", None, None)
    calls.clear()
    with pytest.raises(dl.urllib.error.HTTPError):
        dl.with_retries(missing)
    assert len(calls) == 1


def test_lfs_fetch_lists_only_lfs_pointers(tmp_path):
    path = os.path.join(ROOT, "scripts", "datasets", "lfs_fetch.py")
    lfs = importlib.util.module_from_spec(importlib.util.spec_from_file_location("lfs_fetch", path))
    lfs.__spec__.loader.exec_module(lfs)
    d = tmp_path / "datasets" / "noise"
    d.mkdir(parents=True)
    oid = "ab" * 32
    (d / "a.wav").write_text(f"version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize 320078\n")
    (d / "b.wav").write_bytes(b"RIFF" + bytes(60))                   # already fetched (small but not a pointer)
    os.symlink(d / "a.wav", d / "c.wav")                               # links are not followed
    assert lfs.pointers(str(tmp_path), ["datasets/noise"]) == [(str(d / "a.wav"), oid, 320078)]


def test_throttle_spaces_writes_at_the_limit(monkeypatch):
    slept = []
    monkeypatch.setattr(dl.time, "sleep", slept.append)                # the clock stands still: sleeps accumulate
    monkeypatch.setitem(dl.WRITE_LIMIT, "next", 0.0)
    monkeypatch.setitem(dl.WRITE_LIMIT, "rate", 0.0)
    dl.throttle(10 ** 9)
    assert slept == [] or max(slept) == 0.0
    monkeypatch.setitem(dl.WRITE_LIMIT, "rate", 10e6)
    for _ in range(5):
        dl.throttle(10 ** 7)                                           # 50 MB at 10 MB/s
    assert abs(slept[-1] - 5.0) < 0.1


def test_fetch_verifies_the_checksum(server, tmp_path):
    dest = str(tmp_path / "f.bin")
    dl.fetch(entry(server), dest, connections=4)
    assert open(dest, "rb").read() == server.payload
    with pytest.raises(RuntimeError, match="mismatch"):
        dl.fetch(entry(server, digest="0" * 64), str(tmp_path / "g.bin"), connections=1)
    assert os.path.exists(str(tmp_path / "g.bin.part.bad"))


def test_split_tar_parts_are_extracted_as_one_archive(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for i in range(3):
            data = os.urandom(50_000)
            info = tarfile.TarInfo(f"top/file{i}.bin")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    raw, parts = buf.getvalue(), []
    for i, a in enumerate(range(0, len(raw), len(raw) // 3 + 1)):
        p = tmp_path / f"x.tgz.part{i:02d}"
        p.write_bytes(raw[a:a + len(raw) // 3 + 1])
        parts.append(str(p))
    used = dl.extract(parts, "x.tgz", str(tmp_path / "out"), keep=False)
    assert sorted(os.listdir(tmp_path / "out" / "top")) == ["file0.bin", "file1.bin", "file2.bin"]
    assert used == parts and not any(os.path.exists(p) for p in parts)


@pytest.mark.skipif(shutil.which("zip") is None, reason="needs the zip command")
def test_split_zip_is_merged_before_extraction(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.bin").write_bytes(os.urandom(300_000))
    (tmp_path / "arc").mkdir()
    subprocess.run(["zip", "-q", "-0", "-s", "100k", str(tmp_path / "arc" / "S.zip"), "a.bin"], cwd=src, check=True)
    arc = tmp_path / "arc"
    names = sorted(os.listdir(arc))
    assert "S.z01" in names and "S.zip" in names
    used = dl.extract([str(arc / "S.zip")], "S.zip", str(tmp_path / "out"), keep=False)
    assert (tmp_path / "out" / "a.bin").read_bytes() == (src / "a.bin").read_bytes()
    assert len(used) == len(names) and os.listdir(arc) == []


def test_extraction_rejects_paths_outside_the_target(tmp_path):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../evil.txt", "x")
    with pytest.raises(RuntimeError, match="unsafe"):
        dl.extract_zip(str(z), str(tmp_path / "out"))
    t = tmp_path / "evil.tar"
    with tarfile.open(t, "w") as tf:
        info = tarfile.TarInfo("../evil.txt")
        info.size = 1
        tf.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(tarfile.FilterError):
        dl.extract_tar([str(t)], str(tmp_path / "out2"))
    assert not (tmp_path / "evil.txt").exists()


def test_every_dataset_has_a_license_and_files():
    for name, d in dl.DATASETS.items():
        assert d["license"] and d["files"] and d["size_gb"] > 0, name
