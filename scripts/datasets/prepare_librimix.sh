#!/bin/bash
# Generate the single-speaker noisy part of Libri2Mix (Cosentino et al., 2020): mix_single = speaker 1 + WHAM! noise,
# 16 kHz, "min" mode, with the official scripts of JorisCos/LibriMix, from the sources downloaded by
#     python -I scripts/datasets/download.py get --root <root> librispeech wham_noise
# Train-360 needs the speed-perturbed WHAM! training noise, which the official augment_train_noise.py writes into
# the WHAM! directory first. Licenses: LibriSpeech CC BY 4.0, WHAM! noise CC BY-NC 4.0.
#
# The scripts need sox and a Python with numpy, pandas, soundfile, pyloudnorm, tqdm and pysndfx
# (`python -m pip install pyloudnorm pysndfx`). pysndfx 0.3.6, the sox wrapper of augment_train_noise.py, reads sox's
# output with np.fromstring, whose binary mode NumPy 2 removed; the official script runs unmodified with
# np.fromstring mapped to np.frombuffer.
#
# Usage: bash scripts/datasets/prepare_librimix.sh <root> [python]
# Result: <root>/librimix/Libri2Mix/wav16k/min/{train-100,train-360,dev,test}/{mix_single,s1,noise} (and s2, which
# the official script writes for every mixture type)
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: $0 <root> [python]"; exit 1; }
root=$(realpath -m "$1")
py=${2:-python}
libri=$root/librispeech/LibriSpeech
wham=$root/wham_noise/wham_noise
out=$root/librimix
code=$out/LibriMix
commit=b898ca375c03e4039cdb3dbe3dd485aad58ec7d1       # JorisCos/LibriMix, 2024-02-09
[ -d "$libri/train-clean-360" ] && [ -d "$wham/tr" ] || { echo "run download.py get librispeech wham_noise first"; exit 1; }
mkdir -p "$out"
if [ ! -d "$code/.git" ]; then
  git clone -q https://github.com/JorisCos/LibriMix "$code"
fi
git -C "$code" checkout -q "$commit"
command -v sox > /dev/null || { echo "sox is needed by augment_train_noise.py"; exit 1; }
"$py" -c "import numpy, pandas, soundfile, pyloudnorm, tqdm, pysndfx" \
  || { echo "missing modules for the LibriMix scripts (see the header of this script)"; exit 1; }
(cd "$code" && "$py" -c '
import multiprocessing, runpy, sys
import numpy as np
np.fromstring = lambda s, dtype=float: np.frombuffer(s, dtype=dtype).copy()
multiprocessing.set_start_method("fork")
sys.argv = ["scripts/augment_train_noise.py"] + sys.argv[1:]
runpy.run_path("scripts/augment_train_noise.py", run_name="__main__")
' --wham_dir "$wham")
# The official script skips subsets whose directory exists. With pandas 3 it can crash (segmentation fault) when it
# frees its metadata tables, after or before writing them, so the outputs decide whether the run is complete.
rc=0
(cd "$code" && "$py" scripts/create_librimix_from_metadata.py --librispeech_dir "$libri" --wham_dir "$wham" \
    --metadata_dir metadata/Libri2Mix --librimix_outdir "$out" --n_src 2 --freqs 16k --modes min --types mix_single) \
  || rc=$?
mix=$out/Libri2Mix/wav16k/min
for s in train-100 train-360 dev test; do
  md=$mix/metadata/mixture_${s}_mix_single.csv
  [ -f "$md" ] || { echo "$s: no metadata (exit status $rc): remove $mix/$s and run again"; exit 1; }
  n=$(( $(wc -l < "$md") - 1 ))
  w=$(ls "$mix/$s/mix_single" | wc -l)
  [ "$n" -eq "$w" ] || { echo "$s: $n metadata rows but $w mixtures: remove $mix/$s and run again"; exit 1; }
  echo "$s: $n mixtures"
done
echo "done: $out/Libri2Mix (exit status of the official script: $rc)"
