# Datasets for training and benchmarking

Scripts to download the public corpora that recent speech-enhancement papers train and test on. Each dataset goes
into its own directory, `<root>/<name>/`. Run the Python downloader with `python -I`: it uses only the standard
library, and what it writes is downloaded data.

```bash
python -I scripts/datasets/download.py list                       # names, sizes, licenses
python -I scripts/datasets/download.py check all                  # reachability and sizes, no download
python -I scripts/datasets/download.py get --root <root> demand wham_noise
bash scripts/datasets/download_dns2020.sh <root> all              # DNS 2020 (Git LFS files, lfs_fetch.py)
```

`download.py get` resumes interrupted downloads, uses 8 parallel range requests per file (`--connections`), checks
sizes and the MD5 (Zenodo) or SHA-256 (Hugging Face) checksums, extracts with path checks and deletes each archive
after its extraction (`--keep_archives` keeps them). `<root>/<name>/SOURCE.json` records the URLs, sizes, checksums
and license; `DONE` marks a complete dataset.

`make_training_lists.py` turns the downloaded corpora into the lists of the multi-corpus training set
(`mixture_dataset.py`, `configs/all_data/`; IMPROVEMENT_PLAN.md, 15) and writes the evaluation lists
`filelists/{dns2020_no_reverb,dns2020_with_reverb,ears_wham_test,libri2mix_test}.csv`. Its docstring lists what
it leaves out so that no evaluation speaker, noise recording or file reaches training.

```bash
python -I scripts/datasets/make_training_lists.py --root <root> --out <root>/_lists   # ~15 min, 16 processes
```

## Benchmarks

Sizes checked on 2026-10-24 against the servers (`check all`).

| Dataset | Use | Download | Script | License |
|---|---|---|---|---|
| VoiceBank+DEMAND | the canonical 16 kHz benchmark (824 test files) | 16.1 GB (48 kHz) | `download.py voicebank_demand` | CC BY 4.0 |
| DNS Challenge 2020 | the CleanUNet / CleanUMamba benchmark: 500 h training set, no_reverb test set | 86 GB | `download_dns2020.sh`, then `prepare_dns2020_trainset.sh` | CC BY 4.0, sources' licenses |
| EARS-WHAM v2 | 107 speakers with emotional and whispered speech, WHAM! noise | 142.3 GB of sources | `download.py ears_wham_sources`, then `prepare_ears_wham.sh` | CC BY-NC 4.0 |
| EARS blind test | real noisy recordings, no references | 1.4 GB | `download.py ears_blind_testset` | CC BY-NC 4.0 |
| URGENT 2024 | validation and non-blind test with references, blind test | 1.7 GB | `download.py urgent2024_official`, then `unpack_urgent2024.py` (parquet to wav) | CC BY-NC-SA 4.0 |
| URGENT 2025 | validation and non-blind test with references, blind test | 1.3 GB | `download.py urgent2025_eval` | challenge terms |
| URGENT 2026, track 1 | validation and non-blind test with labels, blind test | 1.9 GB | `download.py urgent2026_track1_eval` | challenge terms |
| URGENT 2026, track 1 training | pre-simulated training and validation set (without ESD) | 153.9 GB | `download.py urgent2026_track1_presim` | CC BY-NC 4.0 |
| Libri2Mix, mix_single | LibriSpeech speaker + WHAM! noise, 16 kHz | 48.3 GB of sources | `download.py librispeech wham_noise`, then `prepare_librimix.sh` | CC BY 4.0 / CC BY-NC 4.0 |

The URGENT sets mix several distortions (noise, reverberation, bandwidth limitation, clipping, codecs, packet loss,
wind) at 8 to 48 kHz, so a 16 kHz denoiser is compared there out of its training conditions.

## Source corpora (speech, noise, room responses)

| Dataset | Download | Script | License |
|---|---|---|---|
| DEMAND, 16 kHz (17 environments) | 1.8 GB | `download.py demand` | CC BY 4.0 |
| WHAM! noise, 8/16 kHz | 18.2 GB | `download.py wham_noise` | CC BY-NC 4.0 |
| DNS5 noise (AudioSet + Freesound) and RIRs | 40.5 GB (64 GB unpacked) | `download.py dns5_noise_ir` | per source (AudioSet CC BY 4.0, Freesound CC0) |
| DNS5 English read speech (LibriVox) | 109.3 GB (131 GB unpacked; 113 of the 306,054 files are all zeros) | `download.py dns5_read_speech` | CC BY 4.0 |
| LibriSpeech (train-clean-100/360, dev, test) | 30.1 GB | `download.py librispeech` | CC BY 4.0 |
| OpenSLR 28 RIRs and noises | 1.3 GB | `download.py openslr28_rirs` | Apache 2.0 |
| FSD50K | 24.7 GB | `download.py fsd50k` | CC BY 4.0; clips CC0 / BY / BY-NC / Sampling+ |
| Free Music Archive, medium / large | 24.2 / 100.7 GB | `download.py fma_medium` / `fma_large` | per track CC licenses |

VCTK 0.92 and MUSAN are already used by this repository (`prepare_vctk_clean.py`, `data.ext_noise`).

## Not downloadable here

| Dataset | Why |
|---|---|
| WSJ0 / WSJ1 (WSJ0-CHiME3, WSJ0-REVERB, URGENT 2025 WSJ part) | LDC license (LDC93S6A, LDC94S13A) |
| CHiME-3 real and simulated | LDC license (LDC2017S24, USB delivery) |
| TIMIT (TIMIT+NOISE92) | LDC license (LDC93S1) |
| NOISEX-92 | sold on CD; no official download, only unofficial copies |
| AudioSet | YouTube links only; its noise clips come through `dns5_noise_ir` |
| Common Voice | Mozilla Data Collective account and terms; only the latest release is distributed (URGENT used v19.0) |
| ESD | license to sign |
| Multilingual LibriSpeech | public (dl.fbaipublicfiles.com/mls) but 2.6 TB for English (0.7 TB as Opus); not in the manifest |

## Download speed

Measured from this machine on 2026-10-24 with one connection: Hugging Face 23 MB/s, GitHub (releases and LFS)
12 MB/s, FMA 2.7 MB/s, Azure (DNS5) 1.7 MB/s, S3 (WHAM!) 0.9 MB/s, OpenSLR 0.8 MB/s, Zenodo 0.6 MB/s, DataShare
0.1 MB/s. With 8 parallel range requests over 32-256 MB pieces (the default of `download.py`), whole files came
at 4.8 MB/s from OpenSLR and 7.7 MB/s from Azure; short probes reached 18 MB/s on S3. At these rates the DNS 2020
download takes about 2 h, EARS-WHAM sources 3 h, the URGENT 2026 training set 2 h, DNS5 read speech 4-6 h.

For DNS 2020, `git lfs pull` fetched the 65,000 noise clips at 1 MB/s on 2026-10-25 (one HTTP/2 connection for
all transfers, on a link with 10-30 % packet loss); `lfs_fetch.py` with 32 HTTP/1.1 connections reached 10 MB/s on
the same link. Metadata requests and every file piece are retried after network errors (name resolution failures,
timeouts, HTTP 408/429/5xx), and rerunning any command resumes it.
