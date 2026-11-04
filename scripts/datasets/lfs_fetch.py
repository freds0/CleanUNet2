"""Fetch the Git LFS files of a public GitHub repository over parallel HTTP/1.1 connections.

`git lfs pull` sends all its transfers through one HTTP/2 connection, which is slow for tens of thousands of small
files on a lossy link. This script replaces each LFS pointer file under the given paths of a clone (made with
GIT_LFS_SKIP_SMUDGE=1) by its content from media.githubusercontent.com, with one keep-alive connection per worker,
and checks the size and SHA-256 that the pointer records. Interrupted runs resume: files already fetched are no
longer pointers. Standard library only; run with `python -I`.

Usage: python -I scripts/datasets/lfs_fetch.py <clone> <owner/repo> <commit> <path>... [--jobs 32]
"""
import argparse
import hashlib
import http.client
import os
import re
import sys
import threading
import time
import urllib.parse

HOST = "media.githubusercontent.com"
UA = "CleanUNet2-dataset-downloader/1.0 (research use)"
POINTER = re.compile(rb"version https://git-lfs\.github\.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize (\d+)\n")


def pointers(clone, paths):
    """(file, sha256, size) of every LFS pointer file under clone/path (links are not followed)."""
    out = []
    for p in paths:
        for dirpath, _, files in os.walk(os.path.join(clone, p)):
            for f in files:
                full = os.path.join(dirpath, f)
                if not os.path.islink(full) and os.path.getsize(full) < 200:
                    with open(full, "rb") as g:
                        m = POINTER.match(g.read())
                    if m:
                        out.append((full, m.group(1).decode(), int(m.group(2))))
    return sorted(out)


def fetch_one(conn, path, dest, sha, size):
    conn.request("GET", path, headers={"User-Agent": UA})
    r = conn.getresponse()
    if r.status != 200:
        r.read()
        raise RuntimeError(f"HTTP {r.status}")
    h, n, tmp = hashlib.sha256(), 0, dest + ".lfs-part"
    with open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            h.update(b)
            n += len(b)
    if n != size or h.hexdigest() != sha:
        os.remove(tmp)
        raise RuntimeError(f"size {n} / sha256 {h.hexdigest()[:12]} do not match the pointer")
    os.replace(tmp, dest)
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clone")
    ap.add_argument("repo", help="owner/repo on GitHub")
    ap.add_argument("commit")
    ap.add_argument("paths", nargs="+", help="directories of the clone to fetch")
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--retries", type=int, default=30)
    args = ap.parse_args()
    clone = os.path.realpath(args.clone)
    todo = pointers(clone, args.paths)
    total = sum(s for _, _, s in todo)
    print(f"{len(todo)} LFS files, {total / 1e9:.2f} GB to fetch with {args.jobs} connections", flush=True)
    lock = threading.Lock()
    state = dict(files=0, bytes=0, t0=time.time(), last=time.time())
    failed = []
    queue = list(reversed(todo))

    def worker():
        conn = None
        while True:
            with lock:
                if not queue:
                    break
                dest, sha, size = queue.pop()
            rel = os.path.relpath(dest, clone)
            path = f"/media/{args.repo}/{args.commit}/{urllib.parse.quote(rel)}"
            for attempt in range(args.retries):
                try:
                    conn = conn or http.client.HTTPSConnection(HOST, timeout=60)
                    n = fetch_one(conn, path, dest, sha, size)
                    break
                except Exception as err:                  # network errors: new connection, wait, retry
                    if conn:
                        conn.close()
                    conn = None
                    if attempt == args.retries - 1:
                        with lock:
                            failed.append((rel, repr(err)))
                        n = 0
                        break
                    time.sleep(min(60, 5 * (attempt + 1)))
            with lock:
                state["files"] += 1
                state["bytes"] += n
                now = time.time()
                if now - state["last"] > 30:
                    rate = state["bytes"] / (now - state["t0"]) / 1e6
                    print(f"    {state['files']} / {len(todo)} files, {state['bytes'] / 1e9:.2f} / "
                          f"{total / 1e9:.2f} GB, {rate:.1f} MB/s", flush=True)
                    state["last"] = now
        if conn:
            conn.close()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - state["t0"]
    print(f"fetched {state['files'] - len(failed)} files, {state['bytes'] / 1e9:.2f} GB in {elapsed:.0f} s", flush=True)
    if failed:
        for rel, err in failed[:10]:
            print(f"  failed: {rel}: {err}")
        sys.exit(f"{len(failed)} files failed: run again to resume")


if __name__ == "__main__":
    main()
