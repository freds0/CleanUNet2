#!/bin/bash
# DNS Challenge 2020 (Interspeech 2020, Reddy et al.) from the interspeech2020/master branch of
# microsoft/DNS-Challenge, which stores its audio with Git LFS (16 kHz wav):
#   clean   64.8 GB, 65,348 files of 30 s (LibriVox read speech)
#   noise   20.9 GB, 65,302 clips (AudioSet, Freesound, DEMAND)
#   test     0.28 GB: synthetic no_reverb and with_reverb (150 clean/noisy pairs each) and 300 real recordings
#   blind    0.22 GB: blind test set (noisy only)
# The no_reverb synthetic test set is the one CleanUNet and CleanUMamba report. License: CC BY 4.0 for the
# repository; the audio keeps the licenses of its sources (see the repository README).
#
# The clone holds only LFS pointers. Each subset is fetched by lfs_fetch.py, which replaces the pointers by their
# content from media.githubusercontent.com over LFS_JOBS parallel HTTP/1.1 connections (default 32) and checks each
# file's size and SHA-256 (`git lfs pull` sends all transfers through one HTTP/2 connection, which is slow for the
# 65,000 small files of a subset on a lossy link). Files are stored once, without an LFS object cache. A subset
# whose files are all present is skipped, so an interrupted run is resumed by running the script again.
#
# Usage: bash scripts/datasets/download_dns2020.sh <root> [test|blind|noise|clean|all]   (default: test)
# Result: <root>/dns2020/DNS-Challenge/datasets/{clean,noise,test_set,blind_test_set}
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: $0 <root> [test|blind|noise|clean|all]"; exit 1; }
root=$(realpath -m "$1")
what=${2:-test}
repo=$root/dns2020/DNS-Challenge
here=$(dirname "$(realpath "$0")")
commit=70f19285c36cca4df2338f9248775ddc50980c6b      # head of interspeech2020/master (2020-07-22)

case $what in
  test) subsets=(datasets/test_set) ;;
  blind) subsets=(datasets/blind_test_set) ;;
  noise) subsets=(datasets/noise) ;;
  clean) subsets=(datasets/clean) ;;
  all) subsets=(datasets/test_set datasets/blind_test_set datasets/noise datasets/clean) ;;
  *) echo "unknown subset '$what'"; exit 1 ;;
esac

if [ ! -d "$repo/.git" ]; then
  mkdir -p "$root/dns2020"
  GIT_LFS_SKIP_SMUDGE=1 git clone -q --depth 1 -b interspeech2020/master https://github.com/microsoft/DNS-Challenge "$repo"
fi
head=$(git -C "$repo" rev-parse HEAD)
[ "$head" = "$commit" ] || echo "[warning] the branch moved: $head (this script was checked at $commit)"

jobs=${LFS_JOBS:-32}
for s in "${subsets[@]}"; do
  if [ "$(find "$repo/$s" -type f -name '*.wav' -size -200c | wc -l)" -eq 0 ]; then
    echo "[$(date '+%m-%d %H:%M')] $s: complete, skipped"
    continue
  fi
  echo "[$(date '+%m-%d %H:%M')] fetch $s"
  ${PYTHON:-python3} -I "$here/lfs_fetch.py" "$repo" microsoft/DNS-Challenge "$head" "$s" --jobs "$jobs" || true
  left=$(find "$repo/$s" -type f -name '*.wav' -size -200c | wc -l)     # LFS pointers are ~130 bytes
  total=$(find "$repo/$s" -type f -name '*.wav' | wc -l)
  echo "[$(date '+%m-%d %H:%M')] $s: $total wav files, $left still pointers"
  [ "$left" -eq 0 ] || { echo "incomplete: run the script again to resume"; exit 1; }
  rm -rf "$repo/.git/lfs/objects"                       # left by earlier versions that used git lfs pull
done
echo "done: $repo/datasets"
