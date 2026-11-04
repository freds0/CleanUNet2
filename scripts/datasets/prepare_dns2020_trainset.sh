#!/bin/bash
# Synthesize the DNS 2020 training set with the recipe of CleanUNet and CleanUMamba: the official synthesizer of the
# interspeech2020/master branch with total_hours 500, SNR from -5 to 25 dB in 31 levels (the other parameters keep
# their official values; the seeded single-process synthesizer, as in CleanUMamba). Needs the clean and noise
# subsets (download_dns2020.sh <root> clean / noise) and a Python with numpy, soundfile, librosa and pandas.
#
# The output is 500 h each of noisy, clean and noise wav files (about 58 GB each); the noise copies are deleted at
# the end unless --keep_noise is given. The synthesizer is Microsoft's code from the clone, run unmodified with its
# own configuration file written here. It resamples the 188 noise clips that are not at 16 kHz (44.1 to 96 kHz) with
# librosa.resample(y, fs_in, fs_out); librosa 0.10 made the rates keyword-only, so the call is mapped to
# resample(y, orig_sr=fs_in, target_sr=fs_out) (whose default filter is now soxr_hq instead of kaiser_best).
#
# Usage: bash scripts/datasets/prepare_dns2020_trainset.sh <root> [python] [--keep_noise]
# Result: <root>/dns2020/training_set/{noisy,clean}, logs in <root>/dns2020/training_set/logs
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: $0 <root> [python] [--keep_noise]"; exit 1; }
root=$(realpath -m "$1")
py=${2:-python}
keep_noise=${3:-}
repo=$root/dns2020/DNS-Challenge
out=$root/dns2020/training_set
for s in clean noise; do
  n=$(find "$repo/datasets/$s" -type f -name '*.wav' -size +200c 2>/dev/null | wc -l)
  [ "$n" -gt 60000 ] || { echo "datasets/$s is missing or incomplete ($n files): run download_dns2020.sh first"; exit 1; }
done
[ ! -d "$out/noisy" ] || { echo "$out/noisy exists; remove it to synthesize again"; exit 1; }
mkdir -p "$out"
cfg=$out/noisyspeech_synthesizer_cleanunet.cfg
sed -e "s|^total_hours:.*|total_hours: 500|" -e "s|^snr_lower:.*|snr_lower: -5|" -e "s|^snr_upper:.*|snr_upper: 25|" \
    -e "s|^total_snrlevels:.*|total_snrlevels: 31|" \
    -e "s|^noise_dir:.*|noise_dir: $repo/datasets/noise|" -e "s|^speech_dir:.*|speech_dir: $repo/datasets/clean|" \
    -e "s|^noisy_destination:.*|noisy_destination: $out/noisy|" \
    -e "s|^clean_destination:.*|clean_destination: $out/clean|" \
    -e "s|^noise_destination:.*|noise_destination: $out/noise|" -e "s|^log_dir:.*|log_dir: $out/logs|" \
    -e "s|^unit_tests_log_dir:.*|unit_tests_log_dir: $out/unittests_logs|" \
    "$repo/noisyspeech_synthesizer.cfg" > "$cfg"
grep -E '^(total_hours|snr_lower|snr_upper|total_snrlevels|noise_dir|speech_dir|noisy_destination):' "$cfg"
echo "[$(date '+%m-%d %H:%M')] synthesizing 500 h"
(cd "$repo" && "$py" -c '
import runpy, sys
import librosa
resample = librosa.resample
librosa.resample = lambda y, *sr, **kw: resample(y, orig_sr=sr[0], target_sr=sr[1], **kw) if sr else resample(y, **kw)
sys.argv = ["noisyspeech_synthesizer_singleprocess.py"] + sys.argv[1:]
runpy.run_path("noisyspeech_synthesizer_singleprocess.py", run_name="__main__")
' --cfg "$cfg")
[ "$keep_noise" = "--keep_noise" ] || rm -rf "$out/noise"
echo "[$(date '+%m-%d %H:%M')] done: $(ls "$out/noisy" | wc -l) noisy files in $out/noisy"
