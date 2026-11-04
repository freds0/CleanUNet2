"""Unpack the URGENT 2024 official evaluation sets (Hugging Face parquet files, as downloaded by
`download.py get urgent2024_official`) into wav files and a metadata table:

    <root>/urgent2024_official/{validation,nonblind_test,blind_test}/{noisy,clean}/<id>.wav
    <root>/urgent2024_official/<split>/metadata.csv            id, sampling_rate, transcript, snr_dB

The wav bytes are written as stored (no re-encoding; 8 to 48 kHz). The blind test set has no clean references.
Needs pyarrow. Run with `python -I scripts/datasets/unpack_urgent2024.py --root <root>`.
"""
import argparse
import csv
import glob
import os
import re

import pyarrow.parquet as pq

SPLITS = {"validation__validation": "validation", "test_nonblind__test_nonblind": "nonblind_test",
          "test_blind__test_blind": "blind_test"}


def unpack(path, out):
    p = pq.ParquetFile(path)
    meta_cols = [c for c in ("id", "sampling_rate", "transcript", "snr_dB") if c in p.schema_arrow.names]
    audio_cols = [c for c in ("noisy_audio", "clean_audio") if c in p.schema_arrow.names]
    for c in audio_cols:
        os.makedirs(os.path.join(out, c.split("_")[0]), exist_ok=True)
    n = 0
    with open(os.path.join(out, "metadata.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(meta_cols)
        for i in range(p.metadata.num_row_groups):
            for r in p.read_row_group(i).to_pylist():
                if not re.fullmatch(r"[A-Za-z0-9_.-]+", r["id"]) or r["id"].startswith("."):
                    raise ValueError(f"unexpected id {r['id']!r} in {path}")
                for c in audio_cols:
                    b = r[c]["bytes"]
                    if b[:4] != b"RIFF" or b[8:12] != b"WAVE":
                        raise ValueError(f"{r['id']} {c}: not a wav file")
                    with open(os.path.join(out, c.split("_")[0], r["id"] + ".wav"), "wb") as g:
                        g.write(b)
                w.writerow([r[c] for c in meta_cols])
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="the --root of download.py")
    base = os.path.join(ap.parse_args().root, "urgent2024_official")
    files = sorted(glob.glob(os.path.join(base, "archives", "*.parquet")))
    if not files:
        raise SystemExit(f"no parquet files in {base}/archives: run download.py get urgent2024_official first")
    for path in files:
        split = SPLITS[os.path.basename(path).split("-")[0]]
        n = unpack(path, os.path.join(base, split))
        print(f"{split}: {n} utterances -> {os.path.join(base, split)}", flush=True)


if __name__ == "__main__":
    main()
