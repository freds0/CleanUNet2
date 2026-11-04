#!/usr/bin/env python
"""Download public speech-enhancement corpora for training and benchmarking (see scripts/datasets/README.md).

Each dataset goes into its own new directory, ``<root>/<name>/``: archives in ``<root>/<name>/archives/`` and the
extracted files next to them. Downloads resume after an interruption (HTTP Range) and are checked against the size
the server reports and, where the source publishes one, an MD5 (Zenodo) or SHA-256 (Hugging Face) checksum. Archives
are unpacked with path checks (no absolute paths, no "..", no links) and deleted after a successful extraction
unless ``--keep_archives`` is given. ``<root>/<name>/SOURCE.json`` records the URLs, sizes, checksums and license.

The script only uses the standard library and writes downloaded data, so run it with ``python -I``.

Usage:
    python -I scripts/datasets/download.py list
    python -I scripts/datasets/download.py check all                     # reachability and sizes, no download
    python -I scripts/datasets/download.py get --root /data/se demand wham_noise
    python -I scripts/datasets/download.py get --root /data/se demand --only DWASHING   # files matching a regex
"""
import argparse
import hashlib
import http.client
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
import zipfile

UA = "CleanUNet2-dataset-downloader/1.0 (research use)"
CHUNK = 8 << 20
GB = 1e9


# ---------------------------------------------------------------------------------------------------------------
# Sources. A file spec is resolved into (url, filename, size, checksum) entries by resolve().
#   url:     direct link (size from the server)
#   zenodo:  every file of a Zenodo record whose name matches `match` (sizes and MD5 from the Zenodo API)
#   hf:      files of a Hugging Face dataset repo under `prefix` (sizes and SHA-256 from the Hub API)
#   gdrive:  a public Google Drive file (name and size from the server)
# `into` extracts the archive into a subdirectory; `group` names the archive that a set of parts forms when
# concatenated (split tar archives); split zip archives (.z01, ...) are merged with `zip -s 0` before extraction.
# ---------------------------------------------------------------------------------------------------------------
def url(u, name=None, **kw):
    return dict(kind="url", url=u, name=name or u.rsplit("/", 1)[-1], **kw)


def zenodo(record, match, **kw):
    return dict(kind="zenodo", record=record, match=match, **kw)


def hf(repo, prefix="", match=".", **kw):
    return dict(kind="hf", repo=repo, prefix=prefix, match=match, **kw)


def gdrive(file_id, **kw):
    return dict(kind="gdrive", id=file_id, **kw)


DNS5 = "https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_training_dataset"
WHAM_S3 = "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com"
EARS = "https://github.com/facebookresearch/ears_dataset/releases/download"
SLR = "https://www.openslr.org/resources"
FMA = "https://os.unil.cloud.switch.ch/fma"
DNS5_READ_PARTS = ["aa", "ab", "ac", "ad", "ae", "af", "ag", "ah", "ai", "aj", "ak", "al", "am", "an", "ao", "ap",
                   "aq", "ar", "as", "at", "au"]

DATASETS = {
    "voicebank_demand": dict(
        desc="VoiceBank+DEMAND (Valentini-Botinhao), official 48 kHz release: 28- and 56-speaker train sets, test set",
        license="CC BY 4.0", size_gb=16.1, nested_zips=True,
        files=[url("https://datashare.ed.ac.uk/download/DS_10283_2791.zip")]),
    "demand": dict(
        desc="DEMAND noise at 16 kHz, 17 of the 18 environments (SCAFE exists only at 48 kHz)",
        license="CC BY 4.0", size_gb=1.8,
        files=[zenodo(1227121, r"(_16k\.zip|DEMAND\.pdf)$")]),
    "wham_noise": dict(
        desc="WHAM! noise, 8 and 16 kHz, train/cv/tt splits (LibriMix, WHAM!)",
        license="CC BY-NC 4.0", size_gb=18.2,
        files=[url(f"{WHAM_S3}/wham_noise.zip")]),
    "ears_wham_sources": dict(
        desc="EARS speech (107 speakers, 48 kHz) and WHAM!48kHz noise in the layout of the EARS-WHAM v2 generator "
             "(EARS/, WHAM48kHz/); then run scripts/datasets/prepare_ears_wham.sh",
        license="CC BY-NC 4.0 (both)", size_gb=142.3,
        files=[url(f"{EARS}/dataset/p{i:03d}.zip", into="EARS") for i in range(1, 108)]
              + [url(f"{WHAM_S3}/high_res_wham.zip", into="WHAM48kHz")]),
    "ears_blind_testset": dict(
        desc="EARS blind test set (noisy speech only, no references)",
        license="CC BY-NC 4.0", size_gb=1.4,
        files=[url(f"{EARS}/blind_testset/blind_testset.zip")]),
    "dns5_noise_ir": dict(
        desc="DNS Challenge 5 (ICASSP 2023) noise (AudioSet and Freesound clips, fullband) and impulse responses",
        license="per source: AudioSet CC BY 4.0, Freesound CC0, see the DNS-Challenge README", size_gb=40.5,
        unpacked_gb=64,
        files=[url(f"{DNS5}/noise_fullband/datasets_fullband.noise_fullband.audioset_{i:03d}.tar.bz2")
               for i in range(7)]
              + [url(f"{DNS5}/noise_fullband/datasets_fullband.noise_fullband.freesound_{i:03d}.tar.bz2")
                 for i in range(2)]
              + [url(f"{DNS5}/datasets_fullband.impulse_responses_000.tar.bz2")]),
    "dns5_read_speech": dict(
        desc="DNS Challenge 5 English read speech (LibriVox, fullband; 306,054 files, 131 GB unpacked, 113 of them all "
             "zeros), a source of URGENT",
        license="CC BY 4.0", size_gb=109.3, unpacked_gb=131,
        files=[url(f"{DNS5}/Track1_Headset/read_speech.tgz.part{p}", group="read_speech.tgz")
               for p in DNS5_READ_PARTS]),
    "librispeech": dict(
        desc="LibriSpeech train-clean-100/360, dev-clean, test-clean (LibriMix speech)",
        license="CC BY 4.0", size_gb=30.1,
        files=[url(f"{SLR}/12/{s}.tar.gz") for s in ["train-clean-100", "train-clean-360", "dev-clean",
                                                     "test-clean"]]),
    "openslr28_rirs": dict(
        desc="OpenSLR 28 simulated and real RIRs and point-source noises (DNS5 and URGENT RIRs)",
        license="Apache 2.0", size_gb=1.3,
        files=[url(f"{SLR}/28/rirs_noises.zip")]),
    "fsd50k": dict(
        desc="FSD50K sound events (dev and eval audio, ground truth, metadata)",
        license="CC BY 4.0 (dataset); clips CC0 / CC BY / CC BY-NC / CC Sampling+", size_gb=24.7,
        files=[zenodo(4060432, r"^FSD50K\.")]),
    "fma_medium": dict(
        desc="Free Music Archive, medium subset (25,000 30-s clips) and metadata",
        license="tracks under their own CC licenses (metadata CC BY 4.0)", size_gb=24.2,
        files=[url(f"{FMA}/fma_medium.zip"), url(f"{FMA}/fma_metadata.zip")]),
    "fma_large": dict(
        desc="Free Music Archive, large subset (106,574 30-s clips) and metadata",
        license="tracks under their own CC licenses (metadata CC BY 4.0)", size_gb=100.7,
        files=[url(f"{FMA}/fma_large.zip"), url(f"{FMA}/fma_metadata.zip")]),
    "urgent2024_official": dict(
        desc="URGENT 2024 (NeurIPS) official validation, non-blind test (with clean references) and blind test",
        license="CC BY-NC-SA 4.0", size_gb=1.7,
        files=[hf("urgent-challenge/urgent2024_official", prefix="data/", match=r"\.parquet$")]),
    "urgent2025_eval": dict(
        desc="URGENT 2025 (Interspeech) official validation and non-blind test (noisy, clean, metadata) and blind test",
        license="see https://urgent-challenge.github.io/urgent2025/", size_gb=1.3,
        files=[gdrive("1Ip-C5tUNGCssT8KAjHUUoh99jkzRH6nm", into="validation"),
               gdrive("11geBBf24WKN1xT_NasnI4JrmKpqNo8h9", into="validation"),
               gdrive("1CU5QKYOgG4fUuJ8oAC6BEhI9ZDhQYZpF", into="validation"),
               gdrive("1rxV6RgA4LAp2I1EnHsln7wI7-UCP6Qer", into="nonblind_test"),
               gdrive("1RarjxOgWkaDV8EjH_eLX169y89PVa3sg", into="nonblind_test"),
               gdrive("1CfhKjfkkUZ60UEOHnlntcQY2m_9pn1uA", into="nonblind_test"),
               gdrive("1dHvYEGHCf9rsB1q-Cd9rXOeQaa_vjG2u", into="blind_test")]),
    "urgent2026_track1_eval": dict(
        desc="URGENT 2026 (ICASSP) track 1 official validation and non-blind test with labels, blind test (noisy)",
        license="see https://urgent-challenge.github.io/urgent2026/track1/", size_gb=1.9,
        files=[gdrive("1dPezrikPASvS2XfvceBF9VStflx3iVNj", into="validation"),
               gdrive("1lDgJmJkxBsxEV8sF58D9Fyk4aRY7dloV", into="nonblind_test"),
               hf("urgent-challenge/urgent2026_leaderboard", prefix="track1/", match=r"blind_noisy\.zip$",
                  into="blind_test")]),
    "urgent2026_track1_presim": dict(
        desc="URGENT 2026 track 1 pre-simulated training and validation set (without the ESD subset)",
        license="CC BY-NC 4.0", size_gb=153.9,
        files=[hf("lichenda/urgent26_track1_universal_se", match=r"\.tgz\.\d+$", group="urgent26_presim.tgz")]),
}


# ---------------------------------------------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------------------------------------------
def gdrive_url(file_id):
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def request(u, start=None, method="GET"):
    headers = {"User-Agent": UA}
    if start is not None:
        headers["Range"] = f"bytes={start}-"
    return urllib.request.urlopen(urllib.request.Request(u, headers=headers, method=method), timeout=60)


def transient(err):
    """Network errors worth retrying: name resolution and connection failures, timeouts, HTTP 408, 429 and 5xx."""
    if isinstance(err, urllib.error.HTTPError):
        return err.code in (408, 429) or err.code >= 500
    return isinstance(err, (OSError, http.client.HTTPException))


def with_retries(fn, *args, tries=10):
    for attempt in range(tries):
        try:
            return fn(*args)
        except Exception as err:
            if attempt == tries - 1 or not transient(err):
                raise
            wait = min(60, 5 * (attempt + 1))
            print(f"    {err!r}; retry {attempt + 1}/{tries} in {wait} s", flush=True)
            time.sleep(wait)


def get_json(u):
    def get():
        with request(u) as r:
            return json.load(r)
    return with_retries(get)


def probe(u):
    """(size, filename from Content-Disposition) of a URL, with a one-byte range request (HEAD as a fallback)."""
    return with_retries(probe_once, u)


def probe_once(u):
    headers = {"User-Agent": UA, "Range": "bytes=0-0"}
    with urllib.request.urlopen(urllib.request.Request(u, headers=headers), timeout=60) as r:
        name = None
        m = re.search(r'filename="?([^";]+)"?', r.headers.get("Content-Disposition", ""))
        if m:
            name = os.path.basename(m.group(1))
        cr = r.headers.get("Content-Range", "")
        if r.status == 206 and "/" in cr and not cr.endswith("/*"):
            return int(cr.rsplit("/", 1)[1]), name
    with request(u, method="HEAD") as r:
        return int(r.headers.get("Content-Length", -1)), name


def resolve(spec):
    """Expand a file spec into dicts with url, name, size, algo, digest, into, group."""
    extra = {k: spec.get(k) for k in ("into", "group")}
    if spec["kind"] == "url":
        return [dict(url=spec["url"], name=spec["name"], size=None, algo=None, digest=None, **extra)]
    if spec["kind"] == "gdrive":
        return [dict(url=gdrive_url(spec["id"]), name=None, size=None, algo=None, digest=None, **extra)]
    if spec["kind"] == "zenodo":
        rec = get_json(f"https://zenodo.org/api/records/{spec['record']}")
        out = []
        for f in rec["files"]:
            if re.search(spec["match"], f["key"]):
                algo, digest = f["checksum"].split(":", 1)
                out.append(dict(url=f["links"]["self"], name=f["key"], size=f["size"], algo=algo, digest=digest,
                                **extra))
        return sorted(out, key=lambda e: e["name"])
    if spec["kind"] == "hf":
        tree = get_json(f"https://huggingface.co/api/datasets/{spec['repo']}/tree/main?recursive=true")
        out = []
        for f in tree:
            p = f["path"]
            if f["type"] == "file" and p.startswith(spec["prefix"]) and re.search(spec["match"], p):
                lfs = f.get("lfs") or {}
                out.append(dict(url=f"https://huggingface.co/datasets/{spec['repo']}/resolve/main/{p}",
                                name=p[len(spec["prefix"]):].replace("/", "__"), size=f["size"],
                                algo="sha256" if lfs.get("oid") else None, digest=lfs.get("oid"), **extra))
        return sorted(out, key=lambda e: e["name"])
    raise ValueError(spec["kind"])


def resolve_all(name, only=None, probe_sizes=True):
    entries = [e for s in DATASETS[name]["files"] for e in resolve(s)]
    for e in entries:
        if probe_sizes and (e["size"] is None or e["name"] is None):
            size, cd_name = probe(e["url"])
            e["size"] = size if e["size"] is None else e["size"]
            e["name"] = e["name"] or cd_name or re.sub(r"\W", "_", e["url"])[-60:]
        e["name"] = os.path.basename(e["name"] or "")          # names come from servers: keep them inside archives/
        if e["name"] in ("", ".", ".."):
            raise RuntimeError(f"bad file name for {e['url']}")
    if only:
        entries = [e for e in entries if re.search(only, e["name"] or e["url"])]
    return entries


# ---------------------------------------------------------------------------------------------------------------
# Download and verification
# ---------------------------------------------------------------------------------------------------------------
def file_digest(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                return h.hexdigest()
            h.update(b)


def supports_range(u):
    def ask():
        with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": UA, "Range": "bytes=0-0"}),
                                    timeout=60) as r:
            return r.status == 206
    try:
        return with_retries(ask)
    except Exception:
        return False


def fetch_segmented(e, part, n, retries):
    """Download a file of known size with n parallel range requests into a preallocated .part file. The progress of
    each segment is kept in <part>.json, so an interrupted download resumes. Single connections to S3, Azure,
    OpenSLR or Zenodo run at 1-2 MB/s here; 8 connections reach 4-18 MB/s."""
    size, state = e["size"], part + ".json"
    if os.path.exists(state) and os.path.exists(part) and os.path.getsize(part) == size:
        with open(state) as f:
            segs = json.load(f)["segments"]
    else:
        # Pieces of 32-256 MB shared by n workers: all connections stay busy until the end, so a slow connection
        # delays only its piece
        step = min(max(32 << 20, -(-size // (4 * n))), 256 << 20)
        segs = [[s, min(s + step, size), 0] for s in range(0, size, step)]       # [start, end, bytes done]
        with open(part, "wb") as f:
            f.truncate(size)
    lock, errors = threading.Lock(), []
    clock = dict(t0=time.time(), last=time.time(), done0=sum(s[2] for s in segs))

    def save():
        with open(state + ".tmp", "w") as f:
            json.dump(dict(size=size, segments=segs), f)
        os.replace(state + ".tmp", state)

    def worker(seg, fd):
        for attempt in range(retries):
            pos = seg[0] + seg[2]
            if pos >= seg[1]:
                return
            try:
                req = urllib.request.Request(e["url"], headers={"User-Agent": UA, "Range": f"bytes={pos}-{seg[1] - 1}"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    if r.status != 206:
                        raise RuntimeError(f"range request answered with HTTP {r.status}")
                    while seg[0] + seg[2] < seg[1]:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        b = b[:seg[1] - seg[0] - seg[2]]
                        os.pwrite(fd, b, seg[0] + seg[2])
                        with lock:
                            seg[2] += len(b)
                            now = time.time()
                            if now - clock["last"] > 30:
                                done = sum(s[2] for s in segs)
                                rate = (done - clock["done0"]) / (now - clock["t0"]) / 1e6
                                print(f"    {e['name']}: {done / GB:.2f} / {size / GB:.2f} GB, {rate:.1f} MB/s",
                                      flush=True)
                                save()
                                clock["last"] = now
            except Exception as err:                       # network errors: wait and resume the segment
                time.sleep(min(60, 5 * (attempt + 1)))
                if attempt == retries - 1:
                    errors.append(repr(err))

    todo = [s for s in segs if s[2] < s[1] - s[0]]

    def pool(fd):
        while True:
            with lock:
                if not todo:
                    return
                seg = todo.pop(0)
            worker(seg, fd)

    fd = os.open(part, os.O_WRONLY)
    try:
        threads = [threading.Thread(target=pool, args=(fd,), daemon=True) for _ in range(min(n, len(todo)))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        os.close(fd)
        with lock:
            save()
    if any(s[0] + s[2] < s[1] for s in segs):
        raise RuntimeError(f"{e['name']}: incomplete after {retries} attempts per segment: {errors[:3]}")
    os.remove(state)


def fetch(e, dest, connections=8, retries=30):
    """Download e["url"] to dest through a .part file (resumable); verify size and checksum."""
    if os.path.exists(dest) and os.path.getsize(dest) == e["size"]:
        return
    part = dest + ".part"
    if e["size"] and e["size"] > (64 << 20) and connections > 1 and supports_range(e["url"]):
        fetch_segmented(e, part, connections, retries)
    else:
        fetch_stream(e, part, retries)
    size = os.path.getsize(part)
    if e["size"] and size != e["size"]:
        raise RuntimeError(f"{e['name']}: size {size} != expected {e['size']} (kept {part})")
    if e["algo"]:
        got = file_digest(part, e["algo"])
        if got != e["digest"]:
            os.replace(part, part + ".bad")
            raise RuntimeError(f"{e['name']}: {e['algo']} mismatch ({got} != {e['digest']}); moved to .bad")
    os.replace(part, dest)


def fetch_stream(e, part, retries):
    """Single-connection download into part, resumed with a Range request after an interruption."""
    for attempt in range(retries):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if e["size"] and have == e["size"]:
            break
        try:
            with request(e["url"], start=have or None) as r:
                if have and r.status != 206:      # the server ignored the range: start over
                    have = 0
                t0, n0, last = time.time(), have, time.time()
                with open(part, "ab" if have else "wb") as f:
                    while True:
                        b = r.read(CHUNK)
                        if not b:
                            break
                        f.write(b)
                        have += len(b)
                        if time.time() - last > 30:
                            rate = (have - n0) / max(time.time() - t0, 1e-6) / 1e6
                            print(f"    {e['name']}: {have / GB:.2f} / {(e['size'] or 0) / GB:.2f} GB, "
                                  f"{rate:.1f} MB/s", flush=True)
                            last = time.time()
            if not e["size"] or have >= e["size"]:
                break
        except (urllib.error.URLError, OSError, TimeoutError) as err:
            wait = min(60, 5 * (attempt + 1))
            print(f"    {e['name']}: {err!r}; retry {attempt + 1}/{retries} in {wait} s", flush=True)
            time.sleep(wait)
    else:
        raise RuntimeError(f"download failed after {retries} attempts: {e['url']}")


# ---------------------------------------------------------------------------------------------------------------
# Safe extraction
# ---------------------------------------------------------------------------------------------------------------
WRITE_LIMIT = dict(rate=0.0, next=0.0)


def throttle(nbytes):
    """Sleep after writing nbytes so that extraction stays under --max_write_mb_s (sustained writes of hundreds of GB
    stalled the DRAM-less NVMe drives of this machine). Idle time does not build up credit."""
    if WRITE_LIMIT["rate"]:
        now = time.time()
        WRITE_LIMIT["next"] = max(WRITE_LIMIT["next"], now) + nbytes / WRITE_LIMIT["rate"]
        time.sleep(max(0.0, WRITE_LIMIT["next"] - now))


def inside(base, target):
    base = os.path.realpath(base)
    return os.path.commonpath([base, os.path.realpath(target)]) == base


def extract_zip(path, dest):
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if (info.external_attr >> 16) & 0o170000 == 0o120000:     # symbolic link
                print(f"    skip link {info.filename}")
                continue
            if os.path.isabs(info.filename) or ".." in info.filename.replace("\\", "/").split("/"):
                raise RuntimeError(f"unsafe path in {path}: {info.filename}")
            if not inside(dest, os.path.join(dest, info.filename)):
                raise RuntimeError(f"unsafe path in {path}: {info.filename}")
            if info.filename.startswith("__MACOSX/") or "/__MACOSX/" in info.filename:
                continue                                              # macOS resource forks (._* files)
            target = os.path.join(dest, info.filename)
            if not info.is_dir() and os.path.isfile(target) and os.path.getsize(target) == info.file_size:
                continue                                              # left by an interrupted extraction
            zf.extract(info, dest)
            throttle(info.file_size)


class Concat(io.RawIOBase):
    """Read several files one after the other as a single stream (the parts of a split archive)."""

    def __init__(self, paths):
        self.paths, self.f = list(paths), None

    def readable(self):
        return True

    def readinto(self, buf):
        while True:
            if self.f is None:
                if not self.paths:
                    return 0
                self.f = open(self.paths.pop(0), "rb")
            n = self.f.readinto(buf)
            if n:
                return n
            self.f.close()
            self.f = None


def extract_tar(paths, dest):
    stream = io.BufferedReader(Concat(paths), buffer_size=CHUNK)
    with tarfile.open(fileobj=stream, mode="r|*") as tf:
        for m in tf:                             # the "data" filter rejects absolute paths, "..", devices, outside links
            tf.extract(m, dest, set_attrs=not m.isdir(), filter="data")
            throttle(m.size)


def is_tar(name):
    return re.search(r"\.(tar|tgz|tar\.gz|tar\.bz2|tbz2|tar\.xz)$", name) is not None


def extract(paths, name, dest, keep):
    """Extract one archive (or the concatenated parts of one) into dest and delete the inputs unless keep. Returns
    the input files it used (empty for a file that is not an archive, such as a parquet or a pdf)."""
    if name.endswith(".zip"):
        splits = sorted(p for p in os.listdir(os.path.dirname(paths[0]))
                        if re.fullmatch(re.escape(name[:-4]) + r"\.z\d\d", p))
        os.makedirs(dest, exist_ok=True)
        print(f"  extract {name} -> {dest}", flush=True)
        if splits:                                  # split zip: merge the parts first
            merged = paths[0][:-4] + ".merged.zip"
            subprocess.run(["zip", "-q", "-s", "0", paths[0], "--out", merged], check=True)
            extract_zip(merged, dest)
            os.remove(merged)
            paths = paths + [os.path.join(os.path.dirname(paths[0]), s) for s in splits]
        else:
            extract_zip(paths[0], dest)
    elif is_tar(name) or len(paths) > 1:
        os.makedirs(dest, exist_ok=True)
        print(f"  extract {name} -> {dest}", flush=True)
        extract_tar(paths, dest)
    else:
        return []
    if not keep:
        for p in paths:
            os.remove(p)
    return paths


def extract_nested_zips(dest, keep):
    """Extract the zip files found inside an extracted archive (the DataShare bundle holds one zip per set)."""
    for dirpath, _, files in os.walk(dest):
        for fn in sorted(files):
            if fn.endswith(".zip") and "archives" not in dirpath.split(os.sep):
                p = os.path.join(dirpath, fn)
                extract([p], fn, os.path.dirname(p), keep)


# ---------------------------------------------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------------------------------------------
def cmd_list(_):
    print(f"{'name':26s} {'GB':>7s}  license / description")
    for name, d in DATASETS.items():
        print(f"{name:26s} {d['size_gb']:7.1f}  {d['license']}\n{'':36s}{d['desc']}")
    print("\nDNS Challenge 2020 (Interspeech): scripts/datasets/download_dns2020.sh (Git LFS, 86 GB).")


def cmd_check(args):
    names = list(DATASETS) if args.names == ["all"] else args.names
    total = 0
    for name in names:
        try:
            entries = resolve_all(name, args.only)
            size = sum(max(e["size"], 0) for e in entries)
            bad = [e["name"] for e in entries if e["size"] is None or e["size"] < 0]
            total += size
            print(f"{name:26s} ok   {len(entries):4d} files {size / GB:8.2f} GB"
                  + (f"  (no size for {bad})" if bad else ""), flush=True)
        except Exception as err:                          # report and go on with the other datasets
            print(f"{name:26s} FAIL {err!r}", flush=True)
    print(f"{'total':26s}           {total / GB:14.2f} GB")


def cmd_get(args):
    os.makedirs(args.root, exist_ok=True)
    WRITE_LIMIT["rate"] = args.max_write_mb_s * 1e6
    for name in args.names:
        d = DATASETS[name]
        base = os.path.join(args.root, name)
        arch = os.path.join(base, "archives")
        os.makedirs(arch, exist_ok=True)
        if os.path.exists(os.path.join(base, "DONE")) and not args.only:
            print(f"{name}: already done")
            continue
        entries = resolve_all(name, args.only)
        todo = [e for e in entries if not os.path.exists(os.path.join(arch, e["name"]) + ".extracted")]
        need = sum(e["size"] for e in todo)
        # Peak use: the archives plus the extracted files (audio barely compresses, except the bz2 archives of DNS5:
        # unpacked_gb). Without --keep_archives each archive is deleted after its extraction, so only the largest
        # one counts twice.
        groups = {}
        for e in todo:
            groups[e["group"] or e["name"]] = groups.get(e["group"] or e["name"], 0) + e["size"]
        unpacked = d["unpacked_gb"] * GB * need / (d["size_gb"] * GB) if d.get("unpacked_gb") else need
        peak = unpacked + (need if args.keep_archives else max(groups.values(), default=0))
        free = shutil.disk_usage(args.root).free
        print(f"{name}: {len(entries)} files, {need / GB:.1f} GB to fetch, peak about {peak / GB:.0f} GB, "
              f"{free / GB:.0f} GB free", flush=True)
        if peak * 1.05 > free and not args.force:
            sys.exit(f"{name}: not enough free space; use --force to try anyway")
        parts = {}
        for i, e in enumerate(entries, 1):
            dest = os.path.join(arch, e["name"])
            if os.path.exists(dest + ".extracted"):
                continue
            print(f"  [{i}/{len(entries)}] {e['name']} ({e['size'] / GB:.2f} GB)", flush=True)
            fetch(e, dest, args.connections)
            out = os.path.join(base, e["into"]) if e["into"] else base
            if e["group"]:
                parts.setdefault(e["group"], (out, []))[1].append(dest)
            elif not args.no_extract:
                for p in extract([dest], e["name"], out, args.keep_archives):
                    open(p + ".extracted", "w").close()
        for gname, (out, paths) in parts.items():
            if not args.no_extract:
                for p in extract(sorted(paths), gname, out, args.keep_archives):
                    open(p + ".extracted", "w").close()
        if d.get("nested_zips") and not args.no_extract:
            extract_nested_zips(base, args.keep_archives)
        with open(os.path.join(base, "SOURCE.json"), "w") as f:
            json.dump(dict(name=name, desc=d["desc"], license=d["license"], date=time.strftime("%Y-%m-%d %H:%M"),
                           files=[{k: e[k] for k in ("url", "name", "size", "algo", "digest")} for e in entries]),
                      f, indent=1)
        if not args.only:
            open(os.path.join(base, "DONE"), "w").close()
        print(f"{name}: done", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    c = sub.add_parser("check")
    c.add_argument("names", nargs="+", help="dataset names or 'all'")
    c.add_argument("--only", help="regex on file names")
    g = sub.add_parser("get")
    g.add_argument("names", nargs="+", choices=list(DATASETS))
    g.add_argument("--root", required=True, help="parent directory; each dataset gets <root>/<name>/")
    g.add_argument("--only", help="regex on file names (download a subset; the dataset is not marked DONE)")
    g.add_argument("--keep_archives", action="store_true")
    g.add_argument("--no_extract", action="store_true", help="download and verify only")
    g.add_argument("--force", action="store_true", help="skip the free-space check")
    g.add_argument("--connections", type=int, default=8, help="parallel range requests per file (1 = one stream)")
    g.add_argument("--max_write_mb_s", type=float, default=0, help="limit the extraction write rate (0: no limit)")
    args = p.parse_args()
    {"list": cmd_list, "check": cmd_check, "get": cmd_get}[args.cmd](args)


if __name__ == "__main__":
    main()
