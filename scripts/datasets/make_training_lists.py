"""Training and evaluation lists for the all-data model (IMPROVEMENT_PLAN.md, 15), from the corpora that download.py
put in --root. Run it with `python -I` (the corpora are downloaded data).

Training lists (TSV, absolute paths, written to --out):
  pairs_<name>.tsv   clean, noisy, sample rate, frames   pre-mixed (clean, noisy) pairs
  speech_<name>.tsv  path, sample rate, frames           clean speech for on-the-fly mixing
  noise_<name>.tsv   path, sample rate, frames           noise for on-the-fly mixing
  rir_<name>.tsv     path, sample rate, frames           room impulse responses
Evaluation lists ("clean|noisy", absolute paths) go to --eval_out (the repository's filelists/).

The task is denoising: every target is the clean (possibly reverberant) speech of its input. What is left out:
  - every evaluation set: VoiceBank-DEMAND test, val (p226, p287) and the 80 speakers of filelists/vctk_unseen.csv
    (so the VoiceBank 56-speaker training set is left out: all its speakers are in vctk_unseen), the DNS 2020 test
    sets, the EARS-WHAM valid/test speakers (p100-p107) and the EARS blind test set, Libri2Mix dev/test, LibriSpeech
    dev/test-clean readers (also in the LibriVox-based DNS read speech, matched by reader id), the URGENT
    validation and test sets;
  - the noise of evaluation sets: WHAM! cv/tt recordings (EARS-WHAM valid/test, Libri2Mix test) and the ten DEMAND
    environments of the VoiceBank-DEMAND test set and of val_unseen;
  - URGENT 2026 pre-simulated pairs with reverberation or any distortion but additive noise (their targets are
    dereverberated or undistorted speech), below 16 kHz, or with an excluded speaker or WHAM! noise;
  - FSD50K clips labelled Human_voice or Singing (foreground voices are not noise); DNS5 read speech files of all
    zeros (dns5_read_speech/ZERO_FILES.txt).
"""

import argparse
import csv
import glob
import multiprocessing as mp
import os
import re
import sys

import soundfile as sf

VB_EVAL_SPEAKERS = {"p226", "p287", "p232", "p257"}               # val and test speakers of VoiceBank-DEMAND
DEMAND_TRAIN_ENVS = ["DKITCHEN", "OMEETING", "PCAFETER", "PRESTO", "PSTATION", "STRAFFIC", "TCAR", "TMETRO"]
FSD_EXCLUDE = {"Human_voice", "Singing"}


def info(path):
    """(path, sample rate, frames) or None for an unreadable / empty file."""
    try:
        i = sf.info(path)
    except Exception:  # noqa: BLE001
        return None
    return (path, i.samplerate, i.frames) if i.frames > 0 else None


def scan(paths, pool, min_seconds=0.0):
    """sf.info of every path; unreadable files and files shorter than min_seconds are dropped."""
    rows = [r for r in pool.imap(info, paths, chunksize=256) if r is not None]
    return [r for r in rows if r[2] >= min_seconds * r[1]]


def write(path, rows):
    with open(path, "w", newline="") as fh:
        csv.writer(fh, delimiter="\t", lineterminator="\n").writerows(rows)
    print(f"{os.path.basename(path)}: {len(rows)} rows, {sum(r[-1] / r[-2] for r in rows) / 3600:.1f} h", flush=True)


def write_eval(path, pairs):
    with open(path, "w") as fh:
        fh.writelines(f"{c}|{n}\n" for c, n in pairs)
    print(f"{os.path.basename(path)}: {len(pairs)} pairs", flush=True)


def by_fileid(clean_dir, noisy_dir):
    """DNS-style pairs: noisy files end in _fileid_<n>.wav, clean files are clean_fileid_<n>.wav."""
    pairs = []
    for noisy in sorted(glob.glob(os.path.join(noisy_dir, "*.wav"))):
        n = re.search(r"_fileid_(\d+)\.wav$", noisy).group(1)
        pairs.append((os.path.join(clean_dir, f"clean_fileid_{n}.wav"), noisy))
    return pairs


def librivox_readers(librispeech, splits):
    """LibriSpeech speaker ids (= LibriVox reader ids) of the given splits."""
    return {d for s in splits for d in os.listdir(os.path.join(librispeech, s)) if d.isdigit()}


def reader_of(name):
    m = re.search(r"reader_(\d+)", name)
    return str(int(m.group(1))) if m else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="/home/fred/DATASETS_CleanUNet")
    p.add_argument("--vb16k", default="/home/fred/DATASETS/VoiceBank-DEMAND-16k")
    p.add_argument("--musan", default="/home/fred/DATASETS/MUSAN/musan")
    p.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    p.add_argument("--out", default="/home/fred/DATASETS_CleanUNet/_lists")
    p.add_argument("--eval_out", default=None, help="default: <repo>/filelists")
    p.add_argument("--jobs", type=int, default=16)
    a = p.parse_args()
    R, eval_out = a.root, a.eval_out or os.path.join(a.repo, "filelists")
    os.makedirs(a.out, exist_ok=True)
    out = lambda name: os.path.join(a.out, name)  # noqa: E731

    with open(os.path.join(a.repo, "filelists", "vctk_unseen.csv")) as fh:
        vctk_unseen = {os.path.basename(line.split("|")[0]).split("_")[0] for line in fh if line.strip()}
    excluded_speakers = vctk_unseen | VB_EVAL_SPEAKERS
    libri = os.path.join(R, "librispeech", "LibriSpeech")
    eval_readers = librivox_readers(libri, ["dev-clean", "test-clean"])
    wham_meta = os.path.join(R, "wham_noise", "wham_noise", "metadata")
    wham_eval_files = set()
    for split in ("cv", "tt"):
        with open(os.path.join(wham_meta, f"noise_meta_{split}.csv")) as fh:
            wham_eval_files |= {row["File ID"] for row in csv.DictReader(fh)}
    print(f"excluded: {len(excluded_speakers)} VCTK speakers, {len(eval_readers)} LibriVox readers, "
          f"{len(wham_eval_files)} WHAM! recordings", flush=True)

    with mp.Pool(a.jobs) as pool:
        # ---- pre-mixed pairs -------------------------------------------------------------------------------------
        ts = os.path.join(R, "dns2020", "training_set")
        pairs = [(c, n) for c, n in by_fileid(os.path.join(ts, "clean"), os.path.join(ts, "noisy"))
                 if reader_of(os.path.basename(n)) not in eval_readers]
        rows = {r[0]: r for r in scan([n for _, n in pairs], pool)}
        write(out("pairs_dns2020_synth.tsv"), [(c, n, *rows[n][1:]) for c, n in pairs if n in rows])

        with open(os.path.join(a.repo, "filelists", "train.csv")) as fh:     # 26 speakers, p226/p287 held out
            pairs = [tuple(os.path.join(a.vb16k, x.strip()) for x in line.split("|")[:2]) for line in fh if line.strip()]
        rows = {r[0]: r for r in scan([n for _, n in pairs], pool)}
        write(out("pairs_vb28.tsv"), [(c, n, *rows[n][1:]) for c, n in pairs if n in rows])

        ew = os.path.join(R, "ears_wham_sources", "EARS-WHAM_v2_16k", "train")
        pairs = []
        for noisy in sorted(glob.glob(os.path.join(ew, "noisy", "p*", "*.wav"))):
            spk, name = noisy.split(os.sep)[-2:]
            pairs.append((os.path.join(ew, "clean", spk, name.split("_")[0] + ".wav"), noisy))
        rows = {r[0]: r for r in scan([n for _, n in pairs], pool)}
        write(out("pairs_ears_wham.tsv"), [(c, n, *rows[n][1:]) for c, n in pairs if n in rows])

        lm = os.path.join(R, "librimix", "Libri2Mix", "wav16k", "min")
        pairs = [(m.replace(f"{os.sep}mix_single{os.sep}", f"{os.sep}s1{os.sep}"), m)
                 for split in ("train-100", "train-360") for m in sorted(glob.glob(os.path.join(lm, split, "mix_single", "*.wav")))]
        rows = {r[0]: r for r in scan([n for _, n in pairs], pool)}
        write(out("pairs_libri2mix.tsv"), [(c, n, *rows[n][1:]) for c, n in pairs if n in rows])

        ur = os.path.join(R, "urgent2026_track1_presim", "urgent26_track2_se_dataset")
        kept = []
        with open(os.path.join(ur, "simulation_train", "log", "meta.tsv")) as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                kinds = {x.split("(")[0] for x in row["augmentation"].split("/")}
                spk = re.match(r"(p\d{3})_", row["speech_uid"])
                if (row["rir_uid"] != "none" or not kinds <= {"none", "", "wind_noise"} or int(row["fs"]) < 16000
                        or (spk and spk.group(1) in excluded_speakers)
                        or row["noise_uid"].split("_")[0] in wham_eval_files
                        or reader_of(row["speech_uid"]) in eval_readers):
                    continue
                kept.append((os.path.join(ur, row["clean_path"]), os.path.join(ur, row["noisy_path"]), int(row["fs"]),
                             int(row["length"])))
        write(out("pairs_urgent26.tsv"), [r for r in kept if os.path.exists(r[1])])

        # ---- clean speech ----------------------------------------------------------------------------------------
        d5 = os.path.join(R, "dns5_read_speech")
        with open(os.path.join(d5, "ZERO_FILES.txt")) as fh:
            zeros = {os.path.basename(line.strip()) for line in fh if line.strip()}
        paths = [x for x in sorted(glob.glob(os.path.join(d5, "mnt", "dnsv5", "clean", "read_speech", "**", "*.wav"),
                                              recursive=True))
                 if os.path.basename(x) not in zeros and reader_of(os.path.basename(x)) not in eval_readers]
        write(out("speech_dns5_read.tsv"), scan(paths, pool, 1.0))

        paths = sorted(glob.glob(os.path.join(libri, "train-clean-*", "*", "*", "*.flac")))
        write(out("speech_librispeech.tsv"), scan(paths, pool, 1.0))

        ears = os.path.join(R, "ears_wham_sources", "EARS")
        paths = sorted(x for x in glob.glob(os.path.join(ears, "p*", "*.wav")) if int(x.split(os.sep)[-2][1:]) < 100)
        write(out("speech_ears.tsv"), scan(paths, pool, 1.0))

        # ---- noise -----------------------------------------------------------------------------------------------
        paths = sorted(glob.glob(os.path.join(R, "dns5_noise_ir", "datasets_fullband", "noise_fullband", "*.wav")))
        write(out("noise_dns5.tsv"), scan(paths, pool, 1.0))

        fsd = os.path.join(R, "fsd50k")
        paths = []
        for split, folder in (("dev", "FSD50K.dev_audio"), ("eval", "FSD50K.eval_audio")):
            with open(os.path.join(fsd, "FSD50K.ground_truth", f"{split}.csv")) as fh:
                for row in csv.DictReader(fh):
                    if not FSD_EXCLUDE & set(row["labels"].split(",")):
                        paths.append(os.path.join(fsd, folder, f"{row['fname']}.wav"))
        write(out("noise_fsd50k.tsv"), scan(sorted(paths), pool, 1.0))

        paths = sorted(glob.glob(os.path.join(R, "wham_noise", "wham_noise", "tr", "*.wav")))
        write(out("noise_wham.tsv"), scan(paths, pool, 1.0))

        paths = sorted(x for env in DEMAND_TRAIN_ENVS for x in glob.glob(os.path.join(R, "demand", env, "ch*.wav")))
        write(out("noise_demand.tsv"), scan(paths, pool, 1.0))

        paths = sorted(glob.glob(os.path.join(a.musan, "noise", "**", "*.wav"), recursive=True)
                       + glob.glob(os.path.join(a.musan, "music", "**", "*.wav"), recursive=True))
        write(out("noise_musan.tsv"), scan(paths, pool, 1.0))

        paths = sorted(glob.glob(os.path.join(R, "fma_large", "fma_large", "*", "*.mp3")))
        write(out("noise_fma.tsv"), scan(paths, pool, 1.0))

        rn = os.path.join(R, "openslr28_rirs", "RIRS_NOISES")
        paths = sorted(glob.glob(os.path.join(rn, "pointsource_noises", "*.wav"))
                       + glob.glob(os.path.join(rn, "real_rirs_isotropic_noises", "*noise*.wav")))
        write(out("noise_slr28.tsv"), scan(paths, pool, 1.0))

        # ---- room impulse responses ------------------------------------------------------------------------------
        paths = sorted(glob.glob(os.path.join(rn, "simulated_rirs", "*", "*", "*.wav"))
                       + [x for x in glob.glob(os.path.join(rn, "real_rirs_isotropic_noises", "*.wav"))
                          if "noise" not in os.path.basename(x)])
        write(out("rir_slr28.tsv"), scan(paths, pool))
        paths = sorted(glob.glob(os.path.join(R, "dns5_noise_ir", "datasets_fullband", "impulse_responses", "SLR26",
                                              "simulated_rirs_48k", "*", "*", "*.wav")))
        write(out("rir_dns5.tsv"), scan(paths, pool))

    # ---- evaluation lists ----------------------------------------------------------------------------------------
    syn = os.path.join(R, "dns2020", "DNS-Challenge", "datasets", "test_set", "synthetic")
    for kind in ("no_reverb", "with_reverb"):
        write_eval(os.path.join(eval_out, f"dns2020_{kind}.csv"),
                   by_fileid(os.path.join(syn, kind, "clean"), os.path.join(syn, kind, "noisy")))
    et = os.path.join(R, "ears_wham_sources", "EARS-WHAM_v2_16k", "test")
    write_eval(os.path.join(eval_out, "ears_wham_test.csv"),
               [(os.path.join(et, "clean", n.split(os.sep)[-2], os.path.basename(n).split("_")[0] + ".wav"), n)
                for n in sorted(glob.glob(os.path.join(et, "noisy", "p*", "*.wav")))])
    write_eval(os.path.join(eval_out, "libri2mix_test.csv"),
               [(m.replace(f"{os.sep}mix_single{os.sep}", f"{os.sep}s1{os.sep}"), m)
                for m in sorted(glob.glob(os.path.join(lm, "test", "mix_single", "*.wav")))])
    return 0


if __name__ == "__main__":
    sys.exit(main())
