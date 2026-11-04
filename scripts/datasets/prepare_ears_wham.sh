#!/bin/bash
# Generate EARS-WHAM v2 (Richter et al., Interspeech 2024; v2 of 2025-04-25) at 16 kHz with the official generator of
# sp-uhh/ears_benchmark, from the sources downloaded by
#     python -I scripts/datasets/download.py get --root <root> ears_wham_sources
# (EARS: 107 speakers, 48 kHz; WHAM!48kHz noise). Splits: speakers p100-p101 valid, p102-p107 test; noise splits
# from WHAM!; SNR -2.5 to 17.5 dB. License: CC BY-NC 4.0.
#
# The generator needs a Python with torch, torchaudio, soundfile, pyloudnorm, numpy and tqdm
# (`python -m pip install pyloudnorm` in the training environment is enough).
#
# Usage: bash scripts/datasets/prepare_ears_wham.sh <root> [python] [--48k]
# Result: <root>/ears_wham_sources/EARS-WHAM_v2_16k/{train,valid,test}/{clean,noisy}
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: $0 <root> [python] [--48k]"; exit 1; }
root=$(realpath -m "$1")
py=${2:-python}
data=$root/ears_wham_sources
code=$root/ears_wham_sources/ears_benchmark
commit=36dc8a88cb2ebf7cc746b51cf2b876bb570bc3e0       # sp-uhh/ears_benchmark, 2025-07-04 (EARS-WHAM v2)
[ -d "$data/EARS" ] && [ -d "$data/WHAM48kHz/high_res_wham" ] || { echo "run download.py get ears_wham_sources first"; exit 1; }
[ "$(ls "$data/EARS" | wc -l)" -eq 107 ] || { echo "EARS is incomplete: $(ls "$data/EARS" | wc -l) of 107 speakers"; exit 1; }
if [ ! -d "$code/.git" ]; then
  git clone -q https://github.com/sp-uhh/ears_benchmark "$code"
fi
git -C "$code" checkout -q "$commit"
"$py" -c "import torch, torchaudio, soundfile, pyloudnorm, tqdm" \
  || { echo "missing modules for the generator (see the header of this script)"; exit 1; }
# The generator reads test_files.json from the current directory and stops if its output directory exists
if [ "${3:-}" = "--48k" ]; then
  (cd "$code" && "$py" generate_ears_wham.py --data_dir "$data")
else
  (cd "$code" && "$py" generate_ears_wham.py --data_dir "$data" --16k)
fi
echo "done: $(ls -d "$data"/EARS-WHAM_v2*)"
