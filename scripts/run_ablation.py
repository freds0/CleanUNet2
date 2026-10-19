#!/usr/bin/env python
"""
Restartable experiment queue for the conditioning / fusion ablation (IMPROVEMENT_PLAN.md).

Every step is skipped when its result already exists (training: marker logs/abl/<name>.done;
evaluation: results/.../summary.json), so the queue can be stopped and relaunched at any time.
Interrupted trainings resume from their last.ckpt. Each training waits for enough free GPU memory
(the GPU is shared) and is retried after out-of-memory errors or when it was killed.
Watchdog: a step whose main thread has exited while other threads (e.g. DataLoader helpers) keep the
process alive shows up as a zombie that is never reaped; it is killed (whole process group) and retried.

Phases:
  0  D0 CleanSpecNet candidates -> B+ ; D1/D2 diagnostics ; D3 dependency of the existing hybrids ;
     K0 control (3 seeds) ; K1 = original waveform FiLM (F0)
  1  injection points P1-P8
  2  signals S2-S5, controls K2 (noisy) / K3 (oracle), spectral-branch quality Q1 (old B) / Q2 (bidirectional)
     -- needs --winner (phase-1 architecture)
  3  operators O1/O3-O7, generator G1/G3, resampling R2/R3, alignment L-1/L+1/Lc  -- needs --winner [--signal]
  4  training of the fusion T1-T8  -- needs --winner [--signal] [--operator]
  C  offline combinations: the winner + the time-domain input estimate (S5) and / or the output fusion (P8)
     -- needs --winner
  5  regimes: Z1 from scratch (30 epochs), Z2 E6  -- needs --winner [--signal] [--operator]
  6  confirmation of --finalists (full protocol, 3 seeds, test set) + K0 with the same protocol;
     --finalists auto = the winner (no lookahead beyond the wave branch) and the best offline configuration
     of phase C  -- auto needs --winner
  N  CleanSpecNet track (section 6 of the plan), protocol PN = stand-alone training + stand-alone evaluation:
     N-a one factor at a time on top of B+ (N1a/N1b front-end, N6a 32 ms window, N7a long crops,
     N7b cumulative normalization, N8a time-domain loss); N-b combined 1-D model, N2 dual-path TF Mamba,
     N3 sub-band, then N4 (phase output) and N4+N5 (phase input) on the best architecture;
     [N-c scale, with --with_nc]; the best spectral branch is retrained (B++) -> results/abl/bplusplus.json
  Nd hybrid check of B++: the best phase-1 fusion points with B++ instead of B+  -- needs --winner
  Nx extras: output fusion P8 with B++; N4 / N4+N5 retrained with the bounded atan2 gradient  -- needs --winner
  U  every run that has a validation result, re-evaluated (same checkpoint) on val_unseen: the validation speakers
     mixed with noise types that VoiceBank-DEMAND never uses (scripts/make_val_unseen.py); comparisons of the
     earlier phases on it and the rank agreement of the deltas on val / val_unseen / test
  R  spectral branch trained with external noise (MUSAN, noise_bank.py): protocol PN settings vs B++'s 30-epoch run,
     the best (mean stand-alone PESQ of val and val_unseen) retrained as Bext; hybrids with Bext trained with the
     same noise against K0 trained with it (3 seeds), on val and val_unseen  -- needs --winner
  RT confirmation of phase R: S5 + P8 without P3 (short protocol), then the full protocol of phase 6 (3 seeds) for K0
     with the external noise and the finalists RX_P3, RX_C3 (+ S5 + P8 when it ties RX_C3), on test, val and
     val_unseen  -- needs --winner and phase R
  V1-V7, VF  next experiments of the plan (section 13), all with Bext and the external noise of phase R, screened on
     val_unseen (val as a guard): V1 training-free blend K0extfull + Bext against RX_C3 (+ RX_P8, the learned output
     fusion alone); V3 external-noise recipe of the waveform branch; V2 condition dropout in RX_C3; V5 fusions
     without extra lookahead; V6 distillation of RX_C3 into the waveform-only model; V4 better spectral branch;
     V7 30 epochs; VF the winners (results/abl/V<n>.json) with the full protocol, 3 seeds, test set  -- need --winner
  W0-W5, WF, WR  the best and the simplest robust system (plan, section 14): W0 low-SNR unseen-noise set;
     W1 training-free (weight soups, output ensembles, test-time augmentation, blend / ensemble weights); W2 loss of
     the waveform branch (compressed magnitude, metric discriminator); W3 distillation of the output ensemble; W4
     recipe of the final waveform-only model; W5 bidirectional spectral branch for the blend; WF final runs
     (30 epochs, 3 seeds, soups, blend, ensemble) on test / val_unseen / val_unseen_low / val; WR robustness report;
     WP polarity fix of the external-noise training (the MUSAN babble makes the models fail on an inverted input)
     -- W1, W3-W5, WF, WR need --winner (RX_C3 is built on it)
  From phase 2 on, the hybrid runs use B++ when results/abl/bplusplus.json exists, else B+.
  --winner auto picks the phase-1 variant with the best validation PESQ.

Usage (detached):
    setsid nohup python scripts/run_ablation.py --phases 0 1 > logs/abl/queue.log 2>&1 &
    python scripts/run_ablation.py --phases N Nd 2 --winner auto --dry_run
"""
import argparse
import json
import os
import shlex
import signal
import statistics
import subprocess
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
STAGE_A = "logs/stageA_cleanunet_mamba/checkpoints/best-epoch016-pesq2.755.ckpt"
STAGE_B_OLD = "logs/stageB_cleanspecnet_mamba/checkpoints/last.ckpt"
HYBRID_C2 = "logs/stageC2_joint/checkpoints/best-epoch009-pesq2.760.ckpt"
HYBRID_DNS = "logs/cleanunet2_mamba/checkpoints/best-epoch012-pesq2.749.ckpt"
E6_DNS = "../cleanumamba/checkpoints/models/CleanUMamba-3N-E6-high.pkl"
LOGS, RESULTS = "logs/abl", "results/abl"
SPEC = "model.architecture.cleanspecnet_params"

# D0: CleanSpecNet candidates (overrides of the architecture / data)
D0_CANDIDATES = {
    "direct_log1p": [],
    "direct_pow": ["model.architecture.spectrogram.compression=pow"],
    "residual_log1p": [f"{SPEC}.output_mode=residual"],
    "residual_pow": [f"{SPEC}.output_mode=residual", "model.architecture.spectrogram.compression=pow"],
    "mask_log1p": [f"{SPEC}.output_mode=mask"],
    "mask_pow": [f"{SPEC}.output_mode=mask", "model.architecture.spectrogram.compression=pow"],
    "mask_pow_seg4s": [f"{SPEC}.output_mode=mask", "model.architecture.spectrogram.compression=pow",
                       "data.segment_size=65536", "data.batch_size=8"],
}

# Track N (IMPROVEMENT_PLAN.md, section 6): CleanSpecNet variants, trained and evaluated alone
N_FRONTENDS = {"N1a_wide_frontend": {"conv_hidden_dim": 256, "num_conv_layers": 2},
               "N1b_linear_frontend": {"frontend": "linear"}}
N_GENERIC = {
    "N6a_win512": ["model.architecture.spectrogram.n_fft=512", "model.architecture.spectrogram.win_length=512"],
    # 6 s crops of 3 joined utterances; 10 epochs ~ the audio of 30 epochs of 2 s crops
    "N7a_concat6s": ["data.concat_utterances=3", "data.segment_size=98304", "data.batch_size=6",
                     "trainer.max_epochs=10"],
    "N7b_cumnorm": [f"{SPEC}.input_norm=cumulative"],
    "N8a_wave_loss": ["loss.spec_terms={l1: 1.0, wave: 0.2}"],
}
N_GAIN = 0.05                         # isolated-PESQ gain over the reference needed to keep an N-a option
PHASE_TERMS = {"l1": 1.0, "phase": 0.3, "complex": 0.1, "consistency": 0.1}
N9_SCALE = {"conv1d": {"N9_deep": {"backbone.n_layers": 8}, "N9_narrow": {"hidden_dim": 256}},
            "tf_mamba": {"N9_deep": {"n_blocks": 6}, "N9_wide": {"channels": 96}},
            "subband": {"N9_deep": {"n_layers": 3}, "N9_wide": {"d_model": 64}}}


def log(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


class Queue:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        os.makedirs(LOGS, exist_ok=True)
        os.makedirs(RESULTS, exist_ok=True)

    # ------------------------------------------------------------ GPU / processes
    @staticmethod
    def free_mib():
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        return int(out.split()[0])

    def wait_gpu(self, need):
        announced = False
        while not self.dry_run:
            free = self.free_mib()
            if free >= need:
                return
            if not announced:
                log(f"waiting for {need} MiB of free GPU memory (free: {free} MiB)")
                announced = True
            time.sleep(60)

    @staticmethod
    def _state(pid):
        try:
            with open(f"/proc/{pid}/stat") as fh:
                return fh.read().rsplit(")", 1)[1].split()[0]
        except OSError:
            return None

    def _run(self, cmd, logfile, zombie_grace=120):
        log("  $ " + " ".join(shlex.quote(c) for c in cmd))
        if self.dry_run:
            return 0, ""
        start = os.path.getsize(logfile) if os.path.exists(logfile) else 0
        with open(logfile, "a") as fh:
            # own process group, so the step and its DataLoader workers can be killed together
            # PYTHONFAULTHANDLER: a native crash (e.g. a segfault) leaves a Python traceback in the log
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True,
                                    env={**os.environ, "PYTHONFAULTHANDLER": "1"})
            zombie_since = None
            while True:
                try:
                    rc = proc.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if self._state(proc.pid) == "Z":
                    zombie_since = zombie_since or time.time()
                    if time.time() - zombie_since > zombie_grace:
                        log(f"  pid {proc.pid}: main thread exited but the process is still alive; killing it")
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        rc = proc.wait()
                        break
                else:
                    zombie_since = None
        with open(logfile, errors="replace") as fh:
            fh.seek(start)
            return rc, fh.read()

    # ------------------------------------------------------------ steps
    def train(self, name, config, overrides, need=10000):
        """Train with train.py; resumes from <ckdir>/last.ckpt. Returns the checkpoint dir (or None)."""
        ckdir = f"{LOGS}/{name}"
        done = f"{LOGS}/{name}.done"
        if os.path.exists(done):
            return ckdir
        cmd = [PY, "train.py", "--config", config, *overrides,
               f"callbacks.best_checkpoint.dirpath={ckdir}", f"resume_from_checkpoint={ckdir}/last.ckpt",
               f"logger.tensorboard.name=abl/{name}"]
        failures = 0
        for attempt in range(1, 11):
            self.wait_gpu(need)
            log(f"start {name} (attempt {attempt})")
            rc, out = self._run(cmd, f"{LOGS}/{name}.log")
            log(f"end {name} exit={rc}")
            if self.dry_run or (rc == 0 and "Training finished" in out):
                if not self.dry_run:
                    open(done, "w").close()
                return ckdir
            if "OutOfMemoryError" in out or "out of memory" in out:
                log(f"{name}: out of GPU memory; retrying from its last checkpoint")
                time.sleep(120)
                continue
            if rc < 0:
                log(f"{name}: killed by signal {-rc}; retrying from its last checkpoint")
                time.sleep(30)
                continue
            # one more try for any other failure (e.g. a transient read error in a DataLoader worker)
            failures += 1
            if failures == 1:
                log(f"{name} FAILED (exit {rc}, not an OOM); retrying once from its last checkpoint")
                time.sleep(30)
                continue
            log(f"{name} FAILED again (not an OOM); see {LOGS}/{name}.log")
            return None
        return None

    def evaluate(self, name, ckpt, out_dir, args=(), need=4000):
        if os.path.exists(f"{out_dir}/summary.json"):
            return True
        if not self.dry_run and not os.path.exists(ckpt):
            log(f"skip eval {name}: no checkpoint {ckpt}")
            return False
        self.wait_gpu(need)
        log(f"eval {name} -> {out_dir}")
        rc, _ = self._run([PY, "evaluate.py", "--checkpoint", ckpt, "--out_dir", out_dir, *args],
                          f"{LOGS}/eval_{name}.log")
        return rc == 0

    def script(self, name, cmd, marker, need=4000):
        if os.path.exists(marker):
            return True
        self.wait_gpu(need)
        log(f"run {name}")
        rc, _ = self._run([PY, *cmd], f"{LOGS}/{name}.log")
        return rc == 0


# ------------------------------------------------------------------ helpers
def arch_path(name):
    return f"../model/fusion/{name}.yaml"


def read_arch(name):
    """Fusion settings of a variant architecture (follows `base:`)."""
    sys.path.insert(0, ROOT)
    from config_utils import load_yaml
    return load_yaml(os.path.join(ROOT, "configs/model/fusion", f"{name}.yaml"))


def spec_branch(q):
    """Spectral branch of the hybrid runs: B++ (track N) when it exists, else B+."""
    path = f"{RESULTS}/bplusplus.json"
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return bplus(q)


def spec_tag(spec):
    return spec.get("tag", "Bplus")


def spec_arch(spec):
    for o in spec["overrides"]:
        if o.startswith(f"{SPEC}="):
            return (yaml.safe_load(o.split("=", 1)[1]) or {}).get("architecture", "conv1d")
    return "conv1d"


def spec_epochs(spec):
    return int(spec.get("epochs", 100))


def bidir_overrides(spec):
    """Overrides turning a spectral branch into its offline (non-causal) version, or None if unsupported."""
    arch = spec_arch(spec)
    if arch == "conv1d":
        return [f"{SPEC}.causal=false", f"{SPEC}.backbone.bidirectional=true"]
    if arch == "tf_mamba":
        return [f"{SPEC}.causal=false"]
    return None


def bplus(q):
    """B+ selection written by phase 0: {'name', 'overrides', 'checkpoint'}."""
    path = f"{RESULTS}/bplus.json"
    if not os.path.exists(path):
        if q.dry_run:
            return {"name": "mask_pow", "overrides": D0_CANDIDATES["mask_pow"],
                    "checkpoint": f"{LOGS}/Bplus/last.ckpt"}
        raise SystemExit("results/abl/bplus.json not found: run phase 0 first")
    with open(path) as fh:
        return json.load(fh)


def arch_overrides_of(ovr):
    """Overrides that define the model (not the data / loss / schedule of the spectral-branch training)."""
    return [o for o in ovr if o.startswith("model.architecture.")]


def variant(q, vid, arch, arch_ovr=(), pc1_ovr=(), pc2_ovr=(), seed=1, skip_pc1=False, spec=None,
            pc1_cfg="configs/ablation/pc1.yaml", pc2_cfg="configs/ablation/pc2.yaml", need=10000, test=False,
            splits=("val",)):
    """Short protocol PC-1 -> PC-2 -> evaluation (normal + condition from another file) on the validation set(s)."""
    spec = spec or spec_branch(q)
    spec_ovr = arch_overrides_of(spec["overrides"])
    common = [f"model.architecture={arch_path(arch)}", *spec_ovr, *arch_ovr, f"seed={seed}"]
    record = f"{RESULTS}/{vid}/variant.json"
    if not q.dry_run and not os.path.exists(record):     # keep the record of the run that was trained
        os.makedirs(f"{RESULTS}/{vid}", exist_ok=True)
        with open(record, "w") as fh:
            json.dump({"id": vid, "arch": arch, "arch_ovr": list(arch_ovr), "pc1_ovr": list(pc1_ovr),
                       "pc2_ovr": list(pc2_ovr), "seed": seed, "skip_pc1": skip_pc1, "spec": spec}, fh, indent=2)
    if skip_pc1:
        init = [f"model.init.full_checkpoint=null", f"model.init.cleanunet_checkpoint={STAGE_A}",
                f"model.init.cleanspecnet_checkpoint={spec['checkpoint']}"]
    else:
        ck1 = q.train(f"{vid}_pc1", pc1_cfg,
                      [*common, f"model.init.cleanspecnet_checkpoint={spec['checkpoint']}", *pc1_ovr], need)
        if ck1 is None:
            return
        init = [f"model.init.full_checkpoint={ck1}/last.ckpt"]
    ck2 = q.train(f"{vid}_pc2", pc2_cfg, [*common, *init, *pc2_ovr], need)
    if ck2 is None:
        return
    for split in splits:
        fast = ["--filelist", f"filelists/{split}.csv", "--fast"]
        name = vid if split == "val" else f"{vid}_{split}"
        q.evaluate(name, f"{ck2}/last.ckpt", f"{RESULTS}/{vid}/{split}", fast)
        q.evaluate(f"{name}_shuffle", f"{ck2}/last.ckpt", f"{RESULTS}/{vid}/{split}_shuffle",
                   [*fast, "--condition", "shuffle"])
    if test:
        q.evaluate(f"{vid}_test", f"{ck2}/last.ckpt", f"{RESULTS}/{vid}/test", ["--filelist", "filelists/test.csv"])
        q.evaluate(f"{vid}_test_shuffle", f"{ck2}/last.ckpt", f"{RESULTS}/{vid}/test_shuffle",
                   ["--filelist", "filelists/test.csv", "--fast", "--condition", "shuffle"])


def k0(q, seed, epochs=5, name=None, test=False):
    name = name or f"K0_s{seed}"
    ck = q.train(name, "configs/ablation/k0.yaml", [f"seed={seed}", f"trainer.max_epochs={epochs}"])
    if ck:
        q.evaluate(name, f"{ck}/last.ckpt", f"{RESULTS}/{name}/val", ["--filelist", "filelists/val.csv", "--fast"])
        if test:
            q.evaluate(f"{name}_test", f"{ck}/last.ckpt", f"{RESULTS}/{name}/test", ["--filelist", "filelists/test.csv"])


def base_run(q, w):
    """The fusion run that phases 2-4 vary around: Nd_<w> (with B++) when track N produced B++, else <w>."""
    nd = f"Nd_{w}"
    if spec_tag(spec_branch(q)) == "Bplusplus" and os.path.exists(f"{RESULTS}/{nd}/val/metrics.csv"):
        return nd
    return w


def compare(q, name, runs, refs=("K0_s1", "K0_s2", "K0_s3"), split="val"):
    refs = [f"{RESULTS}/{r}" for r in refs if os.path.exists(f"{RESULTS}/{r}/{split}/metrics.csv")]
    if not refs or q.dry_run:
        return
    runs = [f"{RESULTS}/{r}" for r in runs]
    q._run([PY, "scripts/compare_runs.py", "--ref", *refs, "--runs", *runs, "--split", split,
            "--out", f"{RESULTS}/{name}.md"], f"{LOGS}/{name}.log")
    log(f"comparison written to {RESULTS}/{name}.md")


def summary_of(vid, split="val"):
    """Mean metrics written by evaluate.py for a run (pesq_wb, stoi, ..., params_M), or None."""
    path = f"{RESULTS}/{vid}/{split}/summary.json"
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)["summary"]


def delta_min(refs=("K0_s1", "K0_s2", "K0_s3")):
    """Smallest PESQ difference that counts (plan, section 3.4): max(0.02, 2 sigma of the K0 seeds)."""
    means = [s["pesq_wb"] for s in (summary_of(r) for r in refs) if s]
    return max(0.02, 2 * statistics.stdev(means)) if len(means) > 1 else 0.02


# ------------------------------------------------------------------ phases
def phase0(q):
    # D0: CleanSpecNet candidates (30 epochs each), selection by D1 (PESQ of the CleanSpecNet magnitude
    # with the noisy phase on whole validation utterances), then B+ = winner trained for 100 epochs.
    names = []
    for cand, ovr in D0_CANDIDATES.items():
        if q.train(f"D0_{cand}", "configs/ablation/stageB_plus.yaml", list(ovr), need=6000):
            names.append(cand)
    sel = f"{RESULTS}/D0/diagnostics.json"
    q.script("D0_select", ["scripts/diagnose_spec_branch.py", "--filelist", "filelists/val.csv",
                           "--out", f"{RESULTS}/D0",
                           *[a for c in names for a in ("--spec_ckpt", f"{c}={LOGS}/D0_{c}/last.ckpt")]], sel)
    if not os.path.exists(f"{RESULTS}/bplus.json") and os.path.exists(sel):
        with open(sel) as fh:
            metrics = json.load(fh)["metrics"]
        scores = {c: metrics[f"cleanspecnet_{c}+noisy_phase"]["pesq_wb"] for c in names}
        best = max(scores, key=scores.get)
        log(f"D0 scores (PESQ of CleanSpecNet + noisy phase): {scores} -> B+ = {best}")
        with open(f"{RESULTS}/bplus.json", "w") as fh:
            json.dump({"name": best, "overrides": D0_CANDIDATES[best], "checkpoint": f"{LOGS}/Bplus/last.ckpt",
                       "scores": scores}, fh, indent=2)
    if not os.path.exists(f"{RESULTS}/bplus.json") and not q.dry_run:
        raise SystemExit("B+ could not be selected")
    sp = bplus(q)
    q.train("Bplus", "configs/ablation/stageB_plus.yaml", [*sp["overrides"], "trainer.max_epochs=100"], need=6000)

    # D1 / D2: spectral branch alone and complementarity with stage A
    q.script("D1D2", ["scripts/diagnose_spec_branch.py", "--wave_ckpt", STAGE_A, "--filelist", "filelists/val.csv",
                      "--spec_ckpt", f"B={STAGE_B_OLD}", "--spec_ckpt", f"Bplus={sp['checkpoint']}",
                      "--out", f"{RESULTS}/D1D2"], f"{RESULTS}/D1D2/diagnostics.json")
    # D3: dependency of the already trained hybrids on their condition
    for name, ck in (("C2", HYBRID_C2), ("DNS", HYBRID_DNS)):
        for cond in ("normal", "shuffle", "off"):
            q.evaluate(f"D3_{name}_{cond}", ck, f"{RESULTS}/D3_{name}/val_{cond}",
                       ["--filelist", "filelists/val.csv", "--fast", "--condition", cond])
    # K0 control (3 seeds) and K1 = F0 (original waveform FiLM with the upsampler) with the short protocol
    for seed in (1, 2, 3):
        k0(q, seed)
    variant(q, "K1_F0", "F0_waveform_film", need=20000)
    compare(q, "phase0", ["K1_F0"])


PHASE1 = ["P1_enc_s2", "P2_enc_fine", "P3_bott_pre", "P4_bott_post", "P5_dec_coarse", "P6_dec_fine",
          "P7_skip_gates", "P8_postfilter"]


def phase1(q):
    for p in PHASE1:
        variant(q, p, p)
    compare(q, "phase1", ["K1_F0", *PHASE1])


def _sig(signal):
    return [f"model.architecture.fusion.signal={signal}"] if signal else []


def phase2(q, w):
    runs = []
    for vid, sig, pc2 in (("S2_spec", "spec", []), ("S3_mask", "mask", []), ("S4_hidden_mask", "[spec_hidden, mask]", []),
                          ("K2_noisy", "noisy_spec", ["loss.mode=waveform"]),
                          ("K3_oracle", "oracle_spec", ["loss.mode=waveform"])):
        variant(q, f"{vid}@{w}", w, _sig(sig), pc2_ovr=pc2)
        runs.append(f"{vid}@{w}")
    variant(q, "S5_input_estimate", "S5_input_estimate")
    runs.append("S5_input_estimate")
    # Q1: the old stage-B CleanSpecNet (log1p, direct output, 10 epochs); with B++, phase Nd already
    # measured the gain as a function of the branch quality (B+ / runner-up / B++), as the plan says
    if spec_tag(spec_branch(q)) != "Bplusplus":
        variant(q, f"Q1_oldB@{w}", w, spec={"name": "old_B", "overrides": [], "checkpoint": STAGE_B_OLD})
        runs.append(f"Q1_oldB@{w}")
    # Q2: bidirectional (offline) version of the current spectral branch (B++ or B+), same training
    sp = spec_branch(q)
    bidir = bidir_overrides(sp)
    name = f"{spec_tag(sp)}_bidir"
    if bidir and q.train(name, "configs/ablation/stageB_plus.yaml",
                         [*sp["overrides"], *bidir, f"trainer.max_epochs={spec_epochs(sp)}"], need=8000):
        spec = {"name": "bidir", "overrides": [*sp["overrides"], *bidir], "checkpoint": f"{LOGS}/{name}/last.ckpt"}
        variant(q, f"Q2_bidir@{w}", w, spec=spec)
        runs.append(f"Q2_bidir@{w}")
    compare(q, "phase2", [base_run(q, w), *runs])


def phase3(q, w, signal=None):
    arch = read_arch(w)
    points = arch["fusion"]["points"]
    frame_rate = all(p in ("bott_pre", "bott_post", "enc_s256", "dec_s256", "skip_s256") for p in points)
    fine = any(p.startswith(("enc_", "dec_", "skip_")) and p not in ("enc_s256", "dec_s256", "skip_s256")
               for p in points)

    def ops(op, extra=None):
        return [f"model.architecture.fusion.points.{p}.op={op}" for p in points if p != "out"] + \
               [f"model.architecture.fusion.points.{p}.{k}={v}" for p in points if p != "out"
                for k, v in (extra or {}).items()]

    tag = f"{w}" + (f"+{signal}" if signal else "")
    base = _sig(signal)
    todo = [("O1_film", ops("film")), ("O3_adaln", ops("adaln_zero")), ("O4_add", ops("add")),
            ("O5_concat_mlp", ops("concat_mlp")),
            ("G1_linear", ["model.architecture.fusion.trunk.layers=0"]),
            ("G3_trunk512", ["model.architecture.fusion.trunk.channels=512", "model.architecture.fusion.trunk.kernel=5"])]
    if frame_rate:
        todo += [("O6_xattn", ops("xattn", {"window": 8})), ("Lm1_shift", ["model.architecture.fusion.shift=-1"]),
                 ("Lp1_shift", ["model.architecture.fusion.shift=1"])]
    if fine:
        todo += [("R2_linear", ["model.architecture.fusion.resample=linear"]),
                 ("R3_convT", ["model.architecture.fusion.resample=convT"])]
    runs = []
    for vid, ovr in todo:
        variant(q, f"{vid}@{tag}", w, [*base, *ovr])
        runs.append(f"{vid}@{tag}")
    variant(q, "O7_exchange", "O7_exchange")
    runs.append("O7_exchange")
    # Lc: causal STFT (needs a CleanSpecNet trained with the causal STFT)
    sp = spec_branch(q)
    causal = ["model.architecture.spectrogram.alignment=causal"]
    name = f"{spec_tag(sp)}_causal"
    if q.train(name, "configs/ablation/stageB_plus.yaml",
               [*sp["overrides"], *causal, f"trainer.max_epochs={spec_epochs(sp)}"], need=8000):
        spec = {"name": "causal", "overrides": [*sp["overrides"], *causal], "checkpoint": f"{LOGS}/{name}/last.ckpt"}
        variant(q, f"Lc_causal_stft@{tag}", w, base, spec=spec)
        runs.append(f"Lc_causal_stft@{tag}")
    compare(q, "phase3", ([] if signal else [base_run(q, w)]) + runs)


def _final_ovr(w, signal, operator):
    ovr = _sig(signal)
    if operator:
        points = read_arch(w)["fusion"]["points"]
        ovr += [f"model.architecture.fusion.points.{p}.op={operator}" for p in points if p != "out"]
    return ovr


def phase4(q, w, signal=None, operator=None):
    ovr = _final_ovr(w, signal, operator)
    tag = f"{w}" + (f"+{signal}" if signal else "") + (f"+{operator}" if operator else "")
    lr5 = "optim.lr_scale={clean_unet: 0.2, clean_spec_net: 0.2, fusion_hub: 5.0, postfilter: 5.0, input_estimate: 5.0}"
    todo = [("T1_no_warmup", dict(skip_pc1=True)),
            ("T2_warmup3000", dict(pc1_ovr=["trainer.max_steps=3000"])),
            ("T3_fusion_lr5e-4", dict(pc2_ovr=[lr5])),
            ("T4_detach", dict(arch_ovr=["model.architecture.fusion.detach=true"])),
            ("T5_no_spec_loss", dict(pc2_ovr=["loss.weight_spec=0.0"])),
            ("T6_spec_loss0.3", dict(pc2_ovr=["loss.weight_spec=0.3"])),
            ("T7_small_init", dict(arch_ovr=["model.architecture.fusion.zero_init=small"])),
            ("T8_oracle_curriculum", dict(pc1_ovr=["fusion_curriculum={oracle_prob: 0.5, steps: 1000}"]))]
    runs = []
    for vid, kw in todo:
        kw = dict(kw)
        kw["arch_ovr"] = [*ovr, *kw.get("arch_ovr", [])]
        variant(q, f"{vid}@{tag}", w, **kw)
        runs.append(f"{vid}@{tag}")
    compare(q, "phase4", ([] if signal or operator else [base_run(q, w)]) + runs)


OFFLINE = ("S5_input_estimate", "Nd_P8_postfilter")     # offline candidates measured alone (phases 2 and Nx)
COMBOS = ("C1_S5", "C2_P8", "C3_S5_P8")


def phaseC(q, w):
    """Offline combinations around the winner, with B++: + time-domain input estimate (S5), + output fusion (P8),
    + both. The iSTFT / STFT of S5 and P8 add lookahead, so these only count for the offline model."""
    s5 = ["model.architecture.fusion.input_estimate=true"]
    p8 = [f"model.architecture.fusion.points.out={_flow(read_arch('P8_postfilter')['fusion']['points']['out'])}"]
    for vid, ovr in zip(COMBOS, (s5, p8, s5 + p8)):
        variant(q, f"{vid}@{w}", w, ovr)
    compare(q, "phaseC", [base_run(q, w), *OFFLINE, *(f"{c}@{w}" for c in COMBOS)])


def best_offline(q, w):
    """Best offline configuration (phase C and its parts) by validation PESQ; among the ones within delta_min of
    the best, the one with the fewest parameters (tie rule of the plan)."""
    cands = [*OFFLINE, *(f"{c}@{w}" for c in COMBOS)]
    scores = {c: s["pesq_wb"] for c, s in ((c, summary_of(c)) for c in cands) if s}
    if not scores:
        if q.dry_run:
            return cands[0]
        raise SystemExit("no offline configuration was evaluated: run phase C first")
    top = max(scores.values())
    tied = [c for c, s in scores.items() if s >= top - delta_min()]
    pick = min(tied, key=lambda c: summary_of(c).get("params_M", float("inf")))
    log("offline configurations (val PESQ): " + ", ".join(
        f"{c} {s:.4f}" for c, s in sorted(scores.items(), key=lambda kv: -kv[1])) + f" -> {pick}")
    return pick


def resolve_finalists(q, finalists, winner):
    """'auto' -> the winner with the current spectral branch (no lookahead beyond the wave branch) and the best
    offline configuration of phase C."""
    if finalists != ["auto"]:
        return finalists
    w = resolve_winner(winner)
    return [base_run(q, w), best_offline(q, w)]


def phase5(q, w, signal=None, operator=None):
    ovr = _final_ovr(w, signal, operator)
    sp = spec_branch(q)
    scratch = ["model.init={}", "optim.lr_scale={}", "trainer.max_epochs=30", *arch_overrides_of(sp["overrides"]),
               f"model.architecture=model/fusion/{w}.yaml", *ovr]
    ck = q.train(f"Z1_scratch@{w}", "configs/train.yaml", scratch, need=12000)
    if ck:
        best = [p for p in os.listdir(ck) if p.startswith("best-")] if os.path.isdir(ck) else []
        best.sort(key=lambda p: float(p.rsplit("pesq", 1)[-1][:-len(".ckpt")]))
        ckpt = f"{ck}/{best[-1]}" if best else f"{ck}/last.ckpt"
        q.evaluate(f"Z1_scratch@{w}_test", ckpt, f"{RESULTS}/Z1_scratch@{w}/test", ["--filelist", "filelists/test.csv"])
    # Z2: E6 (12 ms lookahead): stage A-E6 from the DNS E6 checkpoint, causal STFT, short protocol vs control.
    # Stage A-E6 uses the 20 epochs of the E8 stage A (STAGE_A); without the override base.yaml trains for 300.
    ckA = q.train("A_E6", "configs/train_stageA_cleanunet_mamba.yaml",
                  ["model.architecture=model/cleanunet_mamba_e6.yaml", "trainer.max_epochs=20",
                   f"model.init.cleanunet_from_cleanumamba={E6_DNS}"])
    causal = ["model.architecture.spectrogram.alignment=causal"]
    name = f"{spec_tag(sp)}_causal"
    if ckA and q.train(name, "configs/ablation/stageB_plus.yaml",
                       [*sp["overrides"], *causal, f"trainer.max_epochs={spec_epochs(sp)}"], need=8000):
        points = read_arch(w)["fusion"]["points"]
        e6_ovr = [*causal, *ovr, "model.architecture.fusion.points=" + json.dumps(points),
                  f"model.init.cleanunet_checkpoint={ckA}/last.ckpt"]
        spec = {"name": "causal", "overrides": [*sp["overrides"], *causal], "checkpoint": f"{LOGS}/{name}/last.ckpt"}
        variant(q, f"Z2_E6@{w}", "base_e6", e6_ovr, spec=spec)
        name = "K0_E6"
        ck = q.train(name, "configs/ablation/k0.yaml", ["model.architecture=../model/cleanunet_mamba_e6.yaml",
                                                         f"model.init.cleanunet_checkpoint={ckA}/last.ckpt"])
        if ck:
            q.evaluate(name, f"{ck}/last.ckpt", f"{RESULTS}/{name}/val", ["--filelist", "filelists/val.csv", "--fast"])
    # Z1 against the earlier 30-epoch runs from scratch (wave branch only; original hybrid F0), on the test set;
    # Z2 against the E6 control, on the validation set
    z1 = f"{RESULTS}/Z1_scratch@{w}/test"
    if not q.dry_run and os.path.exists(f"{z1}/metrics.csv"):
        q._run([PY, "scripts/compare_runs.py", "--ref", "results/wave_mamba_scratch_test",
                "--runs", "results/hybrid_mamba_scratch_test", z1, "--split", "", "--out", f"{RESULTS}/phase5_Z1.md"],
               f"{LOGS}/phase5_Z1.log")
        log(f"comparison written to {RESULTS}/phase5_Z1.md")
    compare(q, "phase5_Z2", [f"Z2_E6@{w}"], refs=("K0_E6",))


# ------------------------------------------------------------------ track N
def _flow(value):
    """YAML flow representation of a dict / list (for dotted overrides)."""
    return yaml.safe_dump(value, default_flow_style=True, width=10 ** 6).strip()


def _prefixed(prefix, d):
    return [f"{prefix}.{k}={_flow(v) if isinstance(v, (dict, list)) else v}" for k, v in d.items()]


def _epochs_of(ovr, default=30):
    eps = [int(o.split("=", 1)[1]) for o in ovr if o.startswith("trainer.max_epochs=")]
    return eps[-1] if eps else default


def n_eval(q, name, ckpt):
    """Stand-alone evaluation (protocol PN) on the validation set, with stage A for the complementarity."""
    q.script(f"N_{name}_eval", ["scripts/diagnose_spec_branch.py", "--wave_ckpt", STAGE_A,
                                "--filelist", "filelists/val.csv", "--spec_ckpt", f"{name}={ckpt}",
                                "--out", f"{RESULTS}/N/{name}"], f"{RESULTS}/N/{name}/diagnostics.json")


def n_run(q, name, overrides, need=8000):
    """Protocol PN: train a CleanSpecNet alone (stage-B recipe, 30 epochs, LR 5e-4) and evaluate it alone."""
    record = f"{RESULTS}/N/{name}/spec.json"
    if not q.dry_run and not os.path.exists(record):
        os.makedirs(os.path.dirname(record), exist_ok=True)
        with open(record, "w") as fh:
            json.dump({"name": name, "overrides": list(overrides)}, fh, indent=2)
    ck = q.train(f"N_{name}", "configs/ablation/stageB_plus.yaml", list(overrides), need=need)
    if ck:
        n_eval(q, name, f"{ck}/last.ckpt")
    return ck


def n_overrides(name, base):
    if name == "ref":
        return list(base)
    with open(f"{RESULTS}/N/{name}/spec.json") as fh:
        return json.load(fh)["overrides"]


def n_score(name):
    """Best stand-alone PESQ of a spectral branch (noisy phase, or predicted phase when it has a phase head)."""
    path = f"{RESULTS}/N/{name}/diagnostics.json"
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        m = json.load(fh)["metrics"]
    vals = [m[k]["pesq_wb"] for k in (f"cleanspecnet_{name}+noisy_phase", f"cleanspecnet_{name}+pred_phase")
            if k in m and "pesq_wb" in m[k]]
    return max(vals) if vals else None


def phaseN(q, with_nc=False, final_factor=10 / 3):
    sp = bplus(q)
    base = list(sp["overrides"])
    mode = next((o.split("=", 1)[1] for o in base if o.startswith(f"{SPEC}.output_mode=")), "direct")
    non_spec = [o for o in base if not o.startswith(f"{SPEC}.")]      # compression / data options of B+

    # reference: the D0 winner with the same 30-epoch budget as every N variant
    n_eval(q, "ref", f"{LOGS}/D0_{sp['name']}/last.ckpt")
    # N-a: one factor at a time on top of B+
    for name, fe in N_FRONTENDS.items():
        n_run(q, name, base + _prefixed(SPEC, fe))
    for name, ovr in N_GENERIC.items():
        n_run(q, name, base + ovr)
    scores = {n: n_score(n) for n in ["ref", *N_FRONTENDS, *N_GENERIC]}
    ref = scores["ref"] if scores["ref"] is not None else 0.0
    gain = {n: (v - ref if v is not None else None) for n, v in scores.items()}
    keep = [n for n in N_GENERIC if gain[n] is not None and gain[n] >= N_GAIN]
    fronts = [n for n in N_FRONTENDS if gain[n] is not None and gain[n] >= N_GAIN]
    front = max(fronts, key=lambda n: scores[n]) if fronts else None
    generic = [o for n in keep for o in N_GENERIC[n]]
    log(f"N-a scores {scores} -> kept {keep}, front-end {front}")

    # N-b: combined 1-D model, dual-path TF Mamba (N2), sub-band (N3)
    candidates = ["ref"] + keep + ([front] if front else [])
    if len(keep) + (front is not None) >= 2:
        n_run(q, "N_a_combo", base + (_prefixed(SPEC, N_FRONTENDS[front]) if front else []) + generic)
        candidates.append("N_a_combo")
    tf = {"architecture": "tf_mamba", "causal": True, "output_mode": mode, "channels": 64, "n_blocks": 4}
    sb = {"architecture": "subband", "causal": True, "output_mode": mode, "neighbors": 15, "d_model": 32,
          "n_layers": 2, "fullband": {"backbone": {"type": "mamba", "n_layers": 5, "d_state": 64, "d_conv": 4,
                                                   "expand": 2}, **(N_FRONTENDS[front] if front else {})}}
    n_run(q, "N2_tf_mamba", non_spec + [f"{SPEC}={_flow(tf)}"] + generic, need=14000)
    n_run(q, "N3_subband", non_spec + [f"{SPEC}={_flow(sb)}"] + generic, need=14000)
    candidates += ["N2_tf_mamba", "N3_subband"]
    for n in candidates:
        scores[n] = n_score(n)
    scored = [n for n in candidates if scores.get(n) is not None]
    arch_best = max(scored, key=lambda n: scores[n]) if scored else "ref"
    best_ovr = n_overrides(arch_best, base) if (q.dry_run is False or arch_best == "ref") else list(base)

    # N4 (phase output) and N4 + N5 (phase output and phase input) on the best architecture
    terms = dict(PHASE_TERMS)
    if "N8a_wave_loss" in keep:
        terms["wave"] = 0.2
    phase_ovr = [f"{SPEC}.predict_phase=true", f"loss.spec_terms={_flow(terms)}"]
    n_run(q, "N4_phase_out", best_ovr + phase_ovr, need=14000)
    n_run(q, "N45_phase_in_out", best_ovr + phase_ovr + [f"{SPEC}.input_features=mag_phase"], need=14000)
    for n in ("N4_phase_out", "N45_phase_in_out"):
        scores[n] = n_score(n)

    # N-c (optional): scale of the best architecture
    if with_nc:
        scored = [n for n, v in scores.items() if v is not None]
        top = max(scored, key=lambda n: scores[n]) if scored else "ref"
        top_ovr = n_overrides(top, base) if not q.dry_run else list(base)
        arch = spec_arch({"overrides": top_ovr})
        for name, upd in N9_SCALE[arch].items():
            n_run(q, name, top_ovr + _prefixed(SPEC, upd), need=14000)
            scores[name] = n_score(name)

    # B++: the best spectral branch, retrained with ~100/30 of the budget (as B+)
    if q.dry_run:
        log("dry run: B++ selection skipped")
        return
    scored = {n: v for n, v in scores.items() if v is not None}
    best = max(scored, key=scored.get) if scored else "ref"
    log(f"track N scores: {scored} -> best {best}")
    if best == "ref" or scored[best] - ref < N_GAIN:
        out = dict(bplus(q), tag="Bplus", note="track N found no better spectral branch", scores=scored)
    else:
        ovr = n_overrides(best, base)
        epochs = int(round(_epochs_of(ovr) * final_factor))
        name = f"N_{best}_final"
        ck = q.train(name, "configs/ablation/stageB_plus.yaml", ovr + [f"trainer.max_epochs={epochs}"], need=14000)
        if ck is None:
            return
        n_eval(q, f"{best}_final", f"{ck}/last.ckpt")
        out = {"name": best, "tag": "Bplusplus", "overrides": ovr, "checkpoint": f"{ck}/last.ckpt",
               "epochs": epochs, "scores": scored, "final_score": n_score(f"{best}_final")}
    with open(f"{RESULTS}/bplusplus.json", "w") as fh:
        json.dump(out, fh, indent=2)
    log(f"B++ written to {RESULTS}/bplusplus.json: {out['name']}")


def resolve_winner(winner):
    """'auto' -> the phase-1 variant with the best validation PESQ."""
    if winner != "auto":
        return winner
    best, best_v = None, -1.0
    for p in PHASE1:
        path = f"{RESULTS}/{p}/val/summary.json"
        if os.path.exists(path):
            with open(path) as fh:
                v = json.load(fh)["summary"]["pesq_wb"]
            if v > best_v:
                best, best_v = p, v
    if best is None:
        raise SystemExit("--winner auto: no phase-1 result found")
    log(f"--winner auto -> {best} (val PESQ {best_v:.4f})")
    return best


def phaseNd(q, w, top_points=3):
    """Hybrid check of B++: the best phase-1 points with B++, plus the runner-up spectral branch at the winner."""
    if not os.path.exists(f"{RESULTS}/bplusplus.json") and not q.dry_run:
        raise SystemExit("phase Nd needs results/abl/bplusplus.json (run phase N first)")
    bpp = spec_branch(q)
    if spec_tag(bpp) != "Bplusplus" and not q.dry_run:
        log("B++ = B+ (track N found nothing better): phase Nd skipped")
        return
    ranked = []
    for p in PHASE1:
        path = f"{RESULTS}/{p}/val/summary.json"
        if os.path.exists(path):
            with open(path) as fh:
                ranked.append((json.load(fh)["summary"]["pesq_wb"], p))
    points = [w] + [p for _, p in sorted(ranked, reverse=True) if p != w][:top_points - 1]
    runs = []
    for p in points:
        variant(q, f"Nd_{p}", p, spec=bpp)
        runs += [f"Nd_{p}", p]
    scores = {k: v for k, v in (bpp.get("scores") or {}).items() if v is not None and k != bpp.get("name")}
    if scores:
        runner = max(scores, key=scores.get)
        if runner != "ref" and os.path.exists(f"{LOGS}/N_{runner}/last.ckpt"):
            spec = {"name": runner, "tag": "N_runner_up", "overrides": n_overrides(runner, []),
                    "checkpoint": f"{LOGS}/N_{runner}/last.ckpt"}
            variant(q, f"Nd2_{runner}@{w}", w, spec=spec)
            runs.append(f"Nd2_{runner}@{w}")
    compare(q, "phaseNd", runs)


def phaseNx(q, w):
    """Extras after phase Nd: the output fusion (P8) with B++, and N4 / N4+N5 retrained with the bounded atan2
    gradient (the first N4 diverged to NaN in Atan2Backward). Informational: B++ is not re-selected."""
    variant(q, "Nd_P8_postfilter", "P8_postfilter", spec=spec_branch(q))
    compare(q, "phaseNx", [base_run(q, w), "Nd_P8_postfilter", "P8_postfilter"])
    for old, new in (("N4_phase_out", "N4fix_phase_out"), ("N45_phase_in_out", "N45fix_phase_in_out")):
        n_run(q, new, n_overrides(old, []), need=14000)
    log("stand-alone PESQ with the bounded atan2 gradient: " + ", ".join(
        f"{n} {n_score(n)}" for n in ("N1b_linear_frontend", "N4fix_phase_out", "N45fix_phase_in_out")))


SEED_KEYS = ("pesq_wb", "stoi", "si_sdr", "csig", "cbak", "covl")


def seed_table(q, name, groups, split="test"):
    """Mean +- standard deviation over the seeds of each configuration, with the condition dependency."""
    if q.dry_run:
        return

    def cell(values):
        if not values:
            return "-"
        sd = statistics.stdev(values) if len(values) > 1 else float("nan")
        return f"{statistics.mean(values):.4f} ± {sd:.4f}"

    lines = ["| config | seeds | " + " | ".join(SEED_KEYS) + " | dependency (PESQ) | params (M) |",
             "|---|---|" + "---|" * (len(SEED_KEYS) + 2)]
    for label, runs in groups.items():
        sums = [(r, s) for r, s in ((r, summary_of(r, split)) for r in runs) if s]
        if not sums:
            continue
        dep = []
        for r, s in sums:
            shuf = summary_of(r, f"{split}_shuffle")
            if shuf:
                dep.append(s["pesq_wb"] - shuf["pesq_wb"])
        lines.append(f"| {label} | {len(sums)} | " + " | ".join(cell([s[k] for _, s in sums if k in s])
                                                              for k in SEED_KEYS)
                     + f" | {cell(dep)} | {sums[0][1].get('params_M', float('nan')):.2f} |")
    with open(f"{RESULTS}/{name}.md", "w") as fh:
        fh.write(f"Split '{split}': mean ± standard deviation over the seeds\n\n" + "\n".join(lines) + "\n")
    log(f"seed summary written to {RESULTS}/{name}.md")


def phase6(q, finalists):
    """Full protocol (C1 2000 steps + C2 10 epochs), 3 seeds, test set; K0 with 10 epochs as control."""
    refs = [f"K0full_s{seed}" for seed in (1, 2, 3)]
    for seed in (1, 2, 3):
        k0(q, seed, epochs=10, name=refs[seed - 1], test=True)
    groups = {"K0full": refs}
    for vid in finalists:
        with open(f"{RESULTS}/{vid}/variant.json") as fh:
            v = json.load(fh)
        groups[vid] = [f"F_{vid}_s{seed}" for seed in (1, 2, 3)]
        for seed in (1, 2, 3):
            variant(q, f"F_{vid}_s{seed}", v["arch"], v["arch_ovr"], v["pc1_ovr"] + ["trainer.max_steps=2000"],
                    v["pc2_ovr"] + ["trainer.max_epochs=10"], seed=seed, skip_pc1=v["skip_pc1"], spec=v["spec"],
                    test=True)
    runs = [r for vid in finalists for r in groups[vid]]
    compare(q, "phase6_test", runs, refs, split="test")
    compare(q, "phase6_val", runs, refs)
    seed_table(q, "phase6_seeds", groups)


# ------------------------------------------------------------------ phase U
VAL_UNSEEN = "val_unseen"


def spearman(a, b):
    """Rank correlation (no tie correction: the deltas are continuous)."""
    if len(a) < 3:
        return float("nan")

    def ranks(values):
        order = sorted(range(len(values)), key=values.__getitem__)
        out = [0.0] * len(values)
        for rank, i in enumerate(order):
            out[i] = float(rank)
        return out
    return statistics.correlation(ranks(a), ranks(b))


def agreement(q, name, groups, splits):
    """PESQ delta of each run against the mean of its references on every split, sorted by the second split, and
    the rank correlation of the deltas between every pair of splits (does the ranking carry over?)."""
    if q.dry_run:
        return
    rows = []
    for label, refs, runs in groups:
        for r in runs:
            deltas = []
            for sp in splits:
                s, ref = summary_of(r, sp), [summary_of(x, sp) for x in refs]
                deltas.append(s["pesq_wb"] - statistics.mean(x["pesq_wb"] for x in ref) if s and all(ref) else None)
            rows.append((r, label, deltas))
    rows.sort(key=lambda row: -9 if row[2][1] is None else -row[2][1])
    lines = ["| run | reference | " + " | ".join(f"delta {sp}" for sp in splits) + " |",
             "|---|---|" + "---|" * len(splits)]
    lines += [f"| {r} | {label} | " + " | ".join("-" if d is None else f"{d:+.4f}" for d in deltas) + " |"
              for r, label, deltas in rows]
    corr = []
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            pairs = [(d[i], d[j]) for _, _, d in rows if d[i] is not None and d[j] is not None]
            rho = spearman(*map(list, zip(*pairs))) if pairs else float("nan")
            corr.append(f"{splits[i]} vs {splits[j]}: {rho:+.2f} over {len(pairs)} runs")
    with open(f"{RESULTS}/{name}.md", "w") as fh:
        fh.write("PESQ-WB delta of each run against the mean of its reference runs\n\n" + "\n".join(lines)
                 + "\n\nSpearman correlation of the deltas: " + "; ".join(corr) + "\n")
    log(f"agreement written to {RESULTS}/{name}.md")


def phaseU(q, split=VAL_UNSEEN):
    """Every run that has a validation result, re-evaluated with the same checkpoint on another validation set
    (the condition from another file too, where the run has that result), then the comparisons of the earlier
    phases on it and the agreement of the rankings with the validation and test sets."""
    filelist = f"filelists/{split}.csv"
    if not os.path.exists(filelist):
        raise SystemExit(f"{filelist} not found: run scripts/make_val_unseen.py first")
    runs = sorted(r for r in os.listdir(RESULTS) if os.path.exists(f"{RESULTS}/{r}/val/summary.json"))
    k0, k0full = [f"K0_s{s}" for s in (1, 2, 3)], [f"K0full_s{s}" for s in (1, 2, 3)]
    final = [r for r in runs if r.startswith("F_")]
    e6 = [r for r in runs if r.startswith("Z2_E6")]
    short = [r for r in runs if r not in (*k0, *k0full, *final, *e6, "K0_E6")]
    fast = ["--filelist", filelist, "--fast"]

    def evaluate(group):
        for run in (r for r in group if r in runs):
            with open(f"{RESULTS}/{run}/val/summary.json") as fh:
                ckpt = json.load(fh)["args"]["checkpoint"]
            q.evaluate(f"{run}_{split}", ckpt, f"{RESULTS}/{run}/{split}", fast)
            if os.path.exists(f"{RESULTS}/{run}/val_shuffle/summary.json"):
                q.evaluate(f"{run}_{split}_shuffle", ckpt, f"{RESULTS}/{run}/{split}_shuffle",
                           [*fast, "--condition", "shuffle"])

    # the phase-6 runs first: their test results tell whether the new set ranks the runs like the test set
    evaluate([*k0full, *final])
    compare(q, f"U_phase6_{split}", final, refs=k0full, split=split)
    evaluate([*k0, *short, "K0_E6", *e6])
    compare(q, f"U_short_{split}", short, refs=k0, split=split)
    compare(q, f"U_phase5_Z2_{split}", e6, refs=("K0_E6",), split=split)
    agreement(q, f"U_agreement_{split}", [("K0", k0, short), ("K0full", k0full, final), ("K0_E6", ["K0_E6"], e6)],
              ["val", split, "test"])


# ------------------------------------------------------------------ phase R
MUSAN = "/home/fred/DATASETS/MUSAN/musan"
# share of the training crops whose noise comes from MUSAN (noise_bank.py defaults: noise 0.6, babble of 3-7
# talkers 0.25, music 0.15), at an SNR drawn uniformly in [-5, 20] dB against the P.56 speech level
EXT_SETTINGS = {"ext50": {"dir": MUSAN, "p": 0.5, "snr_db": [-5, 20]},
                "ext80": {"dir": MUSAN, "p": 0.8, "snr_db": [-5, 20]}}


def ext_overrides(setting):
    # the external noise is read and mixed in the DataLoader workers
    return [f"data.ext_noise={_flow(setting)}", "data.num_workers=16"]


def r_eval(q, name, ckpt, split):
    """Stand-alone evaluation of a spectral branch (protocol PN) on a validation split."""
    if split == "val":
        return n_eval(q, name, ckpt)
    out = f"{RESULTS}/N/{name}_{split}"
    q.script(f"N_{name}_eval_{split}", ["scripts/diagnose_spec_branch.py", "--wave_ckpt", STAGE_A,
                                        "--filelist", f"filelists/{split}.csv", "--spec_ckpt", f"{name}={ckpt}",
                                        "--out", out], f"{out}/diagnostics.json")


def r_metrics(name, split):
    path = f"{RESULTS}/N/{name}" + ("" if split == "val" else f"_{split}") + "/diagnostics.json"
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)["metrics"]


def r_score(name, split):
    m = r_metrics(name, split)
    key = f"cleanspecnet_{name}+noisy_phase"
    return m[key]["pesq_wb"] if m and key in m else None


def r_table(q, name, specs, splits=("val", VAL_UNSEEN)):
    """Stand-alone PESQ of each spectral branch (noisy phase) and of its training-free blend with the waveform
    branch (|Y_wave|^(1-w) |S|^w, waveform phase), on every split."""
    if q.dry_run:
        return
    cols = [("cleanspecnet_{}+noisy_phase", "alone"), ("blend_w0.25_{}", "blend w=0.25"),
            ("blend_w0.5_{}", "blend w=0.5")]
    lines = ["| spectral branch | " + " | ".join(f"{label} ({sp})" for _, label in cols for sp in splits) + " |",
             "|---|" + "---|" * (len(cols) * len(splits))]
    wave = {}
    for s in specs:
        cells = []
        for key, _ in cols:
            for sp in splits:
                m = r_metrics(s, sp) or {}
                v = m.get(key.format(s), {}).get("pesq_wb")
                cells.append("-" if v is None else f"{v:.3f}")
                if "wave_branch" in m:
                    wave[sp] = m["wave_branch"]["pesq_wb"]
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    with open(f"{RESULTS}/{name}.md", "w") as fh:
        fh.write("PESQ-WB. Waveform branch alone (stage A): "
                 + ", ".join(f"{sp} {v:.3f}" for sp, v in wave.items()) + "\n\n" + "\n".join(lines) + "\n")
    log(f"spectral-branch table written to {RESULTS}/{name}.md")


def phaseR(q, w):
    """Step 3, a spectral branch that generalizes to unseen noise. R1: the B++ architecture trained with external
    noise (MUSAN, noise_bank.py) under protocol PN, against the same 30-epoch run without it; the best setting by
    the mean stand-alone PESQ of val and val_unseen is retrained for the B++ budget (R2: Bext, results/abl/bext.json).
    R3: the hybrids with Bext trained with the same external noise against K0 trained with it (3 seeds), and P3 with
    Bext on the VoiceBank-DEMAND data alone against K0, on val and val_unseen."""
    sp = spec_branch(q)
    if spec_tag(sp) != "Bplusplus":
        raise SystemExit("phase R starts from B++ (results/abl/bplusplus.json, track N)")
    base, ref = list(sp["overrides"]), sp["name"]
    bpp = os.path.basename(os.path.dirname(sp["checkpoint"]))[len("N_"):]     # B++ itself (100 epochs)
    splits = ("val", VAL_UNSEEN)
    r_eval(q, ref, f"{LOGS}/N_{ref}/last.ckpt", VAL_UNSEEN)
    r_eval(q, bpp, sp["checkpoint"], VAL_UNSEEN)
    for tag, setting in EXT_SETTINGS.items():
        ck = n_run(q, f"R_{tag}", [*base, *ext_overrides(setting)])
        if ck:
            r_eval(q, f"R_{tag}", f"{ck}/last.ckpt", VAL_UNSEEN)
    cands = [ref, *(f"R_{t}" for t in EXT_SETTINGS)]
    scores = {}
    for c in cands:
        s = [r_score(c, split) for split in splits]
        if None not in s:
            scores[c] = statistics.mean(s)
    if q.dry_run:
        pick = cands[1]
    elif not scores:
        raise SystemExit("no stand-alone result for phase R")
    else:
        pick = max(scores, key=scores.get)
        log("spectral branch, mean stand-alone PESQ of val and val_unseen: " + ", ".join(
            f"{c} {s:.4f}" for c, s in sorted(scores.items(), key=lambda kv: -kv[1])) + f" -> {pick}")
    if pick == ref:
        r_table(q, "phaseR_spec", [*cands, bpp])
        log("the external noise does not help the spectral branch alone; phase R stops here")
        return
    tag = pick[len("R_"):]
    final = f"{pick}_final"
    overrides = [*base, *ext_overrides(EXT_SETTINGS[tag])]
    ck = n_run(q, final, [*overrides, f"trainer.max_epochs={spec_epochs(sp)}"])
    if not ck:
        return
    r_eval(q, final, f"{ck}/last.ckpt", VAL_UNSEEN)
    r_table(q, "phaseR_spec", [*cands, bpp, final])
    bext = {"name": final, "tag": "Bext", "overrides": overrides, "checkpoint": f"{ck}/last.ckpt",
            "epochs": spec_epochs(sp), "ext_noise": EXT_SETTINGS[tag]}
    if not q.dry_run:
        with open(f"{RESULTS}/bext.json", "w") as fh:
            json.dump(bext, fh, indent=2)

    ext = ext_overrides(EXT_SETTINGS[tag])
    k0ext = [f"K0ext_s{seed}" for seed in (1, 2, 3)]
    for seed, name in zip((1, 2, 3), k0ext):
        ck0 = q.train(name, "configs/ablation/k0.yaml", [f"seed={seed}", *ext])
        for split in (splits if ck0 else ()):
            q.evaluate(name if split == "val" else f"{name}_{split}", f"{ck0}/last.ckpt", f"{RESULTS}/{name}/{split}",
                       ["--filelist", f"filelists/{split}.csv", "--fast"])
    s5 = ["model.architecture.fusion.input_estimate=true"]
    p8 = [f"model.architecture.fusion.points.out={_flow(read_arch('P8_postfilter')['fusion']['points']['out'])}"]
    hybrids = [(f"RX_P3@{w}", w, []), ("RX_S5", "S5_input_estimate", []), (f"RX_C3@{w}", w, s5 + p8)]
    for vid, arch, ovr in hybrids:
        variant(q, vid, arch, ovr, pc1_ovr=ext, pc2_ovr=ext, spec=bext, splits=splits)
    variant(q, f"RX_P3vb@{w}", w, spec=bext, splits=splits)
    for split in splits:
        compare(q, f"phaseR_{split}", [v for v, _, _ in hybrids], refs=k0ext, split=split)
        compare(q, f"phaseR_vb_{split}", [f"RX_P3vb@{w}", *k0ext], split=split)


def mean_delta(run, refs, splits):
    """(mean over `splits`, per-split list) of the PESQ difference between a run and the mean of its reference runs,
    or (None, None) when a result is missing."""
    deltas = []
    for split in splits:
        s, ref = summary_of(run, split), [summary_of(r, split) for r in refs]
        if not s or not all(ref):
            return None, None
        deltas.append(s["pesq_wb"] - statistics.mean(r["pesq_wb"] for r in ref))
    return statistics.mean(deltas), deltas


def phaseRT(q, w):
    """Confirmation of phase R. RT1: S5 + P8 without P3 (short protocol, Bext, external noise), to learn whether
    RX_C3 needs P3. RT2: the full protocol of phase 6 (3 seeds) for K0 trained with the external noise and for the
    finalists, evaluated on the test set, val and val_unseen: RX_P3 (no lookahead beyond the wave branch), RX_C3 (the
    best offline hybrid of phase R) and S5 + P8 when it beats K0ext on both validation sets and its mean delta over
    them is within delta_min of RX_C3's (the same model without P3). The seeds run in turn (seed 1 of every
    configuration first), so an interrupted queue still leaves a complete comparison."""
    path = f"{RESULTS}/bext.json"
    if not os.path.exists(path):
        raise SystemExit("results/abl/bext.json not found: run phase R first")
    with open(path) as fh:
        bext = json.load(fh)
    ext = ext_overrides(bext["ext_noise"])
    splits = ("val", VAL_UNSEEN)
    k0ext = [f"K0ext_s{seed}" for seed in (1, 2, 3)]
    s5 = ["model.architecture.fusion.input_estimate=true"]
    p8 = [f"model.architecture.fusion.points.out={_flow(read_arch('P8_postfilter')['fusion']['points']['out'])}"]
    hybrids = {f"RX_P3@{w}": (w, []), f"RX_C3@{w}": (w, s5 + p8), "RX_S5P8": ("S5_input_estimate", p8)}
    p3, c3, s5p8 = hybrids

    # RT1
    variant(q, s5p8, *hybrids[s5p8], pc1_ovr=ext, pc2_ovr=ext, spec=bext, splits=splits)
    for split in splits:
        done = [r for r in (c3, s5p8, "RX_S5", p3) if os.path.exists(f"{RESULTS}/{r}/{split}/metrics.csv")]
        if done:
            compare(q, f"phaseRT_short_{split}", done, refs=k0ext, split=split)
    finalists = [p3, c3]
    m_c3, _ = mean_delta(c3, k0ext, splits)
    m_s5p8, d_s5p8 = mean_delta(s5p8, k0ext, splits)
    if q.dry_run:
        finalists.append(s5p8)
    elif m_c3 is not None and m_s5p8 is not None:
        if min(d_s5p8) > 0 and m_s5p8 >= m_c3 - delta_min(k0ext):
            finalists.append(s5p8)
        log(f"{s5p8}: delta vs K0ext val {d_s5p8[0]:+.4f}, {VAL_UNSEEN} {d_s5p8[1]:+.4f}, mean {m_s5p8:+.4f}; "
            f"{c3} mean {m_c3:+.4f}; delta_min {delta_min(k0ext):.3f}")
    log("finalists: " + ", ".join(finalists))

    # RT2
    k0full = [f"K0extfull_s{seed}" for seed in (1, 2, 3)]
    groups = {vid: [f"F_{vid}_s{seed}" for seed in (1, 2, 3)] for vid in finalists}
    for seed, name in zip((1, 2, 3), k0full):
        ck = q.train(name, "configs/ablation/k0.yaml", [f"seed={seed}", "trainer.max_epochs=10", *ext])
        if ck:
            q.evaluate(f"{name}_test", f"{ck}/last.ckpt", f"{RESULTS}/{name}/test", ["--filelist", "filelists/test.csv"])
            for split in splits:
                q.evaluate(name if split == "val" else f"{name}_{split}", f"{ck}/last.ckpt",
                           f"{RESULTS}/{name}/{split}", ["--filelist", f"filelists/{split}.csv", "--fast"])
        for vid in finalists:
            variant(q, groups[vid][seed - 1], *hybrids[vid], pc1_ovr=[*ext, "trainer.max_steps=2000"],
                    pc2_ovr=[*ext, "trainer.max_epochs=10"], seed=seed, spec=bext, splits=splits, test=True)

    # reports: the finalists against K0 with the same data and protocol on every split, everything against K0full
    # (VoiceBank-DEMAND data only, phase 6) on the test set, and the seed means next to the phase-6 runs
    runs = [r for vid in finalists for r in groups[vid]]
    for split in ("test", *splits):
        done = [r for r in runs if os.path.exists(f"{RESULTS}/{r}/{split}/metrics.csv")]
        if done:
            compare(q, f"phaseRT_{split}", done, refs=k0full, split=split)
    done = [r for r in (*k0full, *runs) if os.path.exists(f"{RESULTS}/{r}/test/metrics.csv")]
    if done:
        compare(q, "phaseRT_vs_K0full_test", done, refs=[f"K0full_s{seed}" for seed in (1, 2, 3)], split="test")
    seeds = {"K0full (VB data)": [f"K0full_s{seed}" for seed in (1, 2, 3)],
             "P3 B++ (VB data)": [f"F_Nd_{w}_s{seed}" for seed in (1, 2, 3)],
             "C3 B++ (VB data)": [f"F_C3_S5_P8@{w}_s{seed}" for seed in (1, 2, 3)],
             "K0extfull": k0full, **groups}
    for split in ("test", *splits):
        seed_table(q, f"phaseRT_seeds_{split}", seeds, split=split)


# ------------------------------------------------------------------ phase V (plan, section 13)
SPLITS_V = ("val", VAL_UNSEEN)
VAL_GUARD = -0.04          # a candidate may not lose more than 2 delta_min on the old validation set
K0EXT = [f"K0ext_s{seed}" for seed in (1, 2, 3)]
K0EXTFULL = [f"K0extfull_s{seed}" for seed in (1, 2, 3)]
V3_RECIPES = {"V3_p30": {"p": 0.3}, "V3_p80": {"p": 0.8}, "V3_snr0": {"snr_db": [0, 20]},
              "V3_noise": {"weights": {"noise": 1.0, "babble": 0.0, "music": 0.0}},
              "V3_babble": {"weights": {"noise": 0.4, "babble": 0.4, "music": 0.2}}}
V2_PROBS = (0.15, 0.3)
V6_WEIGHTS = (0.5, 1.0)


def load_bext():
    path = f"{RESULTS}/bext.json"
    if not os.path.exists(path):
        raise SystemExit("results/abl/bext.json not found: run phase R first")
    with open(path) as fh:
        return json.load(fh)


def s5p8():
    """Overrides that turn the P3 architecture into C3 (P3 + S5 + P8)."""
    return ["model.architecture.fusion.input_estimate=true",
            f"model.architecture.fusion.points.out={_flow(read_arch('P8_postfilter')['fusion']['points']['out'])}"]


def dmin_split(refs, split):
    """delta_min (plan, 3.4) from the spread of the reference seeds on a given split."""
    means = [s["pesq_wb"] for s in (summary_of(r, split) for r in refs) if s]
    return max(0.02, 2 * statistics.stdev(means)) if len(means) > 1 else 0.02


def mean_pesq(runs, split):
    vals = [s["pesq_wb"] for s in (summary_of(r, split) for r in runs) if s]
    return statistics.mean(vals) if len(vals) == len(runs) and vals else None


def write_json(name, obj):
    with open(f"{RESULTS}/{name}.json", "w") as fh:
        json.dump(obj, fh, indent=2)
    log(f"decision written to {RESULTS}/{name}.json: {json.dumps(obj)}")


def read_json(name):
    path = f"{RESULTS}/{name}.json"
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def k0_run(q, name, overrides, splits=SPLITS_V, test=False, need=10000):
    """Waveform-only run with configs/ablation/k0.yaml (stage A + extra training), evaluated on `splits`."""
    ck = q.train(name, "configs/ablation/k0.yaml", list(overrides), need)
    if ck:
        for split in splits:
            q.evaluate(name if split == "val" else f"{name}_{split}", f"{ck}/last.ckpt", f"{RESULTS}/{name}/{split}",
                       ["--filelist", f"filelists/{split}.csv", "--fast"])
        if test:
            q.evaluate(f"{name}_test", f"{ck}/last.ckpt", f"{RESULTS}/{name}/test", ["--filelist", "filelists/test.csv"])
    return ck


def off_evals(q, vid, splits, test=False):
    """Hybrid evaluated with every conditioning path off (does the waveform branch still work alone?)."""
    ck = f"{LOGS}/{vid}_pc2/last.ckpt"
    for split in (*splits, *(("test",) if test else ())):
        q.evaluate(f"{vid}_{split}_off", ck, f"{RESULTS}/{vid}/{split}_off",
                   ["--filelist", f"filelists/{split}.csv", "--fast", "--condition", "off"])


def pesq_table(q, name, runs, splits, conds=("",), note=""):
    """Mean PESQ-WB of each run on every split / condition (normal, _shuffle, _off)."""
    if q.dry_run:
        return
    cols = [(sp, c) for sp in splits for c in conds]
    lines = ["| run | " + " | ".join(f"{sp}{c}" for sp, c in cols) + " |", "|---|" + "---|" * len(cols)]
    for r in runs:
        cells = []
        for sp, c in cols:
            s = summary_of(r, f"{sp}{c}")
            cells.append("-" if not s else f"{s['pesq_wb']:.4f}")
        lines.append(f"| {r} | " + " | ".join(cells) + " |")
    with open(f"{RESULTS}/{name}.md", "w") as fh:
        fh.write(f"PESQ-WB. {note}\n\n" + "\n".join(lines) + "\n")
    log(f"table written to {RESULTS}/{name}.md")


def screen(run, refs, ref_run=None):
    """Delta of a run against the mean of `refs` on val_unseen and val, and whether it passes the screening rule:
    val_unseen delta >= delta_min (of the refs on val_unseen) over the refs, or over `ref_run` when given, and val
    delta >= VAL_GUARD. Returns (passes, d_unseen, d_val)."""
    _, d = mean_delta(run, refs, SPLITS_V)
    if d is None:
        return False, None, None
    d_val, d_unseen = d
    bar = dmin_split(refs, VAL_UNSEEN)
    if ref_run is not None:
        _, d_ref = mean_delta(ref_run, refs, SPLITS_V)
        if d_ref is None:
            return False, d_unseen, d_val
        bar += d_ref[1]
    return d_unseen >= bar and d_val >= VAL_GUARD, d_unseen, d_val


def phaseV1(q, w):
    """V1, ensemble control: the training-free blend of K0extfull (3 seeds) with Bext, w in {0.25, 0.5} picked on
    val_unseen, against K0extfull and RX_C3 (full protocol); the chosen w on the test set. Also RX_P8 (the learned
    output fusion alone, short protocol) against the same blend of the short K0ext runs."""
    bext = load_bext()
    ext = ext_overrides(bext["ext_noise"])

    def blend(run, wgt, split):
        vid = f"V1_{run}_w{wgt:g}"
        args = ["--filelist", f"filelists/{split}.csv", "--blend_spec", bext["checkpoint"], "--blend_w", f"{wgt:g}"]
        if split != "test":
            args.append("--fast")
        q.evaluate(f"{vid}_{split}", f"{LOGS}/{run}/last.ckpt", f"{RESULTS}/{vid}/{split}", args)
        return vid

    weights = (0.25, 0.5)
    for wgt in weights:
        for run in K0EXTFULL:
            for split in SPLITS_V:
                blend(run, wgt, split)
    scores = {wgt: mean_pesq([f"V1_{r}_w{wgt:g}" for r in K0EXTFULL], VAL_UNSEEN) for wgt in weights}
    scores = {k: v for k, v in scores.items() if v is not None}
    wbest = max(scores, key=scores.get) if scores else 0.25
    log(f"V1 blend on {VAL_UNSEEN} (mean of 3 seeds): {scores} -> w = {wbest:g}")
    full = [blend(run, wbest, "test") for run in K0EXTFULL]
    short = []
    for run in K0EXT:
        for split in SPLITS_V:
            vid = blend(run, wbest, split)
        short.append(vid)
    variant(q, "RX_P8", "P8_postfilter", pc1_ovr=ext, pc2_ovr=ext, spec=bext, splits=SPLITS_V)

    c3 = [f"F_RX_C3@{w}_s{seed}" for seed in (1, 2, 3)]
    for split in ("test", *SPLITS_V):
        compare(q, f"V1_{split}", [*full, *c3], refs=K0EXTFULL, split=split)
        compare(q, f"V1_blend_vs_RXC3_{split}", full, refs=c3, split=split)
        seed_table(q, f"V1_seeds_{split}", {"K0extfull": K0EXTFULL, f"blend K0extfull + Bext (w {wbest:g})": full,
                                            "RX_C3": c3}, split=split)
    for split in SPLITS_V:
        compare(q, f"V1_short_{split}", [*short, "RX_P8", f"RX_C3@{w}"], refs=K0EXT, split=split)
    if q.dry_run:
        return
    m_blend, m_c3 = mean_pesq(full, VAL_UNSEEN), mean_pesq(c3, VAL_UNSEEN)
    if m_blend is None or m_c3 is None:
        return
    bar = dmin_split(K0EXTFULL, VAL_UNSEEN)
    write_json("V1", {"w": wbest, "blend_val_unseen": m_blend, "rx_c3_val_unseen": m_c3, "delta_min": bar,
                      "fusion_justified": m_c3 - m_blend >= bar,
                      "blend_test": mean_pesq(full, "test"), "rx_c3_test": mean_pesq(c3, "test")})


def phaseV3(q):
    """V3, external-noise recipe of the waveform branch: K0 (short protocol, seed 1) with one change at a time to
    the Bext recipe (p 0.5, SNR -5..20 dB, noise / babble / music 0.6 / 0.25 / 0.15). The best by val_unseen,
    if it passes the screening rule against K0ext, gets seeds 2 and 3; it is kept if the 3-seed mean still passes."""
    bext = load_bext()
    runs = []
    for name, upd in V3_RECIPES.items():
        k0_run(q, name, ["seed=1", *ext_overrides({**bext["ext_noise"], **upd})])
        runs.append(name)
    for split in SPLITS_V:
        compare(q, f"V3_{split}", runs, refs=K0EXT, split=split)
    if q.dry_run:
        return
    passed = {}
    for name in runs:
        ok, du, dv = screen(name, K0EXT)
        log(f"V3 {name}: delta vs K0ext {VAL_UNSEEN} {du}, val {dv} -> {'pass' if ok else 'no'}")
        if ok:
            passed[name] = du
    winner = None
    if passed:
        best = max(passed, key=passed.get)
        seeds = [best] + [f"{best}_s{seed}" for seed in (2, 3)]
        for seed, name in zip((2, 3), seeds[1:]):
            k0_run(q, name, [f"seed={seed}", *ext_overrides({**bext["ext_noise"], **V3_RECIPES[best]})])
        for split in SPLITS_V:
            compare(q, f"V3_seeds_{split}", seeds, refs=K0EXT, split=split)
        d = [mean_delta(r, K0EXT, SPLITS_V)[1] for r in seeds]
        if all(d):
            du = statistics.mean(x[1] for x in d)
            dv = statistics.mean(x[0] for x in d)
            if du >= dmin_split(K0EXT, VAL_UNSEEN) and dv >= VAL_GUARD:
                winner = best
            log(f"V3 {best}, 3 seeds: delta {VAL_UNSEEN} {du:+.4f}, val {dv:+.4f} -> {'kept' if winner else 'dropped'}")
    write_json("V3", {"winner": winner, "setting": {**bext["ext_noise"], **V3_RECIPES[winner]} if winner else None})


def phaseV2(q, w):
    """V2, condition dropout in RX_C3: with probability p a PC-2 batch runs with every conditioning path off.
    Dependency (condition from another file) and the PESQ with the condition off, against RX_C3 (short protocol).
    Kept: the best p whose val_unseen delta is within delta_min of RX_C3's and whose dependency is below 1."""
    bext = load_bext()
    ext = ext_overrides(bext["ext_noise"])
    rx = f"RX_C3@{w}"
    off_evals(q, rx, SPLITS_V)
    runs = []
    for p in V2_PROBS:
        vid = f"V2_C3_drop{int(round(p * 100))}"
        variant(q, vid, w, s5p8(), pc1_ovr=ext, pc2_ovr=[*ext, f"fusion_dropout={p}"], spec=bext, splits=SPLITS_V)
        off_evals(q, vid, SPLITS_V)
        runs.append(vid)
    for split in SPLITS_V:
        compare(q, f"V2_{split}", [rx, *runs], refs=K0EXT, split=split)
    ref_unseen = mean_pesq(K0EXT, VAL_UNSEEN)
    pesq_table(q, "V2_conditions", [*K0EXT, rx, *runs], SPLITS_V, ("", "_shuffle", "_off"),
               note=f"Normal, condition from another file (_shuffle) and condition off (_off). K0ext {VAL_UNSEEN}: "
                    f"{ref_unseen}")
    if q.dry_run:
        return
    _, d_rx = mean_delta(rx, K0EXT, SPLITS_V)
    bar = dmin_split(K0EXT, VAL_UNSEEN)
    kept = {}
    for vid, p in zip(runs, V2_PROBS):
        _, d = mean_delta(vid, K0EXT, SPLITS_V)
        s, sh = summary_of(vid, VAL_UNSEEN), summary_of(vid, f"{VAL_UNSEEN}_shuffle")
        if d is None or d_rx is None or not s or not sh:
            continue
        dep = s["pesq_wb"] - sh["pesq_wb"]
        ok = d[1] >= d_rx[1] - bar and d[0] >= VAL_GUARD and dep < 1.0
        log(f"V2 {vid}: delta {VAL_UNSEEN} {d[1]:+.4f} (RX_C3 {d_rx[1]:+.4f}), dependency {dep:.3f} -> "
            f"{'pass' if ok else 'no'}")
        if ok:
            kept[vid] = (d[1], p)
    best = max(kept, key=lambda k: kept[k][0]) if kept else None
    # the probability that V5 uses for P3: the kept one, else the one with the best val_unseen delta
    deltas = {p: mean_delta(v, K0EXT, SPLITS_V)[1] for v, p in zip(runs, V2_PROBS)}
    p_v5 = kept[best][1] if best else max((p for p in deltas if deltas[p]), key=lambda p: deltas[p][1],
                                          default=V2_PROBS[-1])
    write_json("V2", {"winner": best, "p": kept[best][1] if best else None, "p_for_v5": p_v5})


def phaseV5(q, w):
    """V5, fusions without lookahead beyond the waveform branch, screened on val_unseen with Bext and the external
    noise (short protocol) against K0ext: other points (P2, P5), the spectrum as signal (S2), the additive operator
    (O4), the linear generator (G1) and P3 with condition dropout. RX_P3 is the reference hybrid."""
    bext = load_bext()
    ext = ext_overrides(bext["ext_noise"])
    v2 = read_json("V2") or {}
    p = v2.get("p_for_v5") or V2_PROBS[-1]
    points = read_arch(w)["fusion"]["points"]
    todo = [("V5_P2", "P2_enc_fine", [], []), ("V5_P5", "P5_dec_coarse", [], []),
            ("V5_S2", w, _sig("spec"), []),
            ("V5_O4", w, [f"model.architecture.fusion.points.{pt}.op=add" for pt in points if pt != "out"], []),
            ("V5_G1", w, ["model.architecture.fusion.trunk.layers=0"], []),
            (f"V5_P3_drop{int(round(p * 100))}", w, [], [f"fusion_dropout={p}"])]
    runs = []
    for vid, arch, ovr, pc2 in todo:
        variant(q, vid, arch, ovr, pc1_ovr=ext, pc2_ovr=[*ext, *pc2], spec=bext, splits=SPLITS_V)
        runs.append(vid)
    for split in SPLITS_V:
        compare(q, f"V5_{split}", [f"RX_P3@{w}", *runs], refs=K0EXT, split=split)
    if q.dry_run:
        return
    winners = []
    for vid, arch, ovr, pc2 in todo:
        ok, du, dv = screen(vid, K0EXT)
        log(f"V5 {vid}: delta vs K0ext {VAL_UNSEEN} {du}, val {dv} -> {'pass' if ok else 'no'}")
        if ok:
            winners.append({"id": vid, "arch": arch, "arch_ovr": ovr, "pc2": pc2, "delta_unseen": du})
    winners.sort(key=lambda x: -x["delta_unseen"])
    write_json("V5", {"winners": winners[:1]})


def phaseV6(q, w):
    """V6, distillation of RX_C3 (full protocol, seed 1, frozen) into the waveform-only model: K0 short protocol with
    the external noise plus weight x waveform loss against the teacher output, against K0ext."""
    bext = load_bext()
    ext = ext_overrides(bext["ext_noise"])
    teacher = f"{LOGS}/F_RX_C3@{w}_s1_pc2/last.ckpt"
    runs = []
    for lam in V6_WEIGHTS:
        name = f"V6_kd{lam:g}"
        k0_run(q, name, ["seed=1", *ext, "distill=" + _flow({"teacher_checkpoint": teacher, "weight": lam})],
               need=14000)
        runs.append(name)
    for split in SPLITS_V:
        compare(q, f"V6_{split}", runs, refs=K0EXT, split=split)
    if q.dry_run:
        return
    passed = {}
    for name, lam in zip(runs, V6_WEIGHTS):
        ok, du, dv = screen(name, K0EXT)
        log(f"V6 {name}: delta vs K0ext {VAL_UNSEEN} {du}, val {dv} -> {'pass' if ok else 'no'}")
        if ok:
            passed[lam] = du
    lam = max(passed, key=passed.get) if passed else None
    write_json("V6", {"winner": f"V6_kd{lam:g}" if lam else None, "weight": lam})


def phaseV4(q, w):
    """V4, a better spectral branch with the external noise (protocol PN, 30 epochs) against R_ext50 (the same
    budget): N3 sub-band, a deeper backbone (8 Mamba layers) and, as a separate offline (whole-utterance) class, the
    bidirectional backbone. A candidate that gains >= N_GAIN of stand-alone val_unseen PESQ is retrained for the
    Bext budget and tried in RX_C3 (short protocol) against RX_C3 with Bext."""
    bext = load_bext()
    ext = ext_overrides(bext["ext_noise"])
    base = list(bext["overrides"])
    if q.dry_run:
        n3 = ["model.architecture.cleanspecnet_params={architecture: subband}"]
    else:
        n3 = [o for o in n_overrides("N3_subband", []) if not o.startswith("data.")]
    cands = {"V4_N3": ([*n3, *ext], 14000), "V4_deep": ([*base, f"{SPEC}.backbone.n_layers=8"], 8000),
             "V4_bidir": ([*base, *(bidir_overrides(bext) or [])], 8000)}
    for name, (ovr, need) in cands.items():
        ck = n_run(q, name, ovr, need=need)
        if ck:
            r_eval(q, name, f"{ck}/last.ckpt", VAL_UNSEEN)
    r_table(q, "V4_spec", ["R_ext50", *cands])
    if q.dry_run:
        return
    ref = r_score("R_ext50", VAL_UNSEEN)
    scores = {n: r_score(n, VAL_UNSEEN) for n in cands}
    log(f"V4 stand-alone {VAL_UNSEEN} PESQ: R_ext50 {ref}, " + ", ".join(f"{n} {s}" for n, s in scores.items()))
    causal = {n: s for n, s in scores.items() if n != "V4_bidir" and s is not None and s - ref >= N_GAIN}
    picks = ([max(causal, key=causal.get)] if causal else []) + \
            (["V4_bidir"] if scores.get("V4_bidir") is not None and scores["V4_bidir"] - ref >= N_GAIN else [])
    hybrids = []
    for name in picks:
        ovr = cands[name][0]
        final = f"{name}_final"
        ck = n_run(q, final, [*ovr, f"trainer.max_epochs={spec_epochs(bext)}"], need=cands[name][1])
        if not ck:
            continue
        r_eval(q, final, f"{ck}/last.ckpt", VAL_UNSEEN)
        spec = {"name": final, "tag": name, "overrides": ovr, "checkpoint": f"{ck}/last.ckpt",
                "epochs": spec_epochs(bext), "ext_noise": bext["ext_noise"]}
        vid = f"{name}_C3"
        variant(q, vid, w, s5p8(), pc1_ovr=ext, pc2_ovr=ext, spec=spec, splits=SPLITS_V)
        hybrids.append((vid, spec))
    r_table(q, "V4_spec", ["R_ext50", "R_ext50_final", *cands, *(f"{n}_final" for n in picks)])
    for split in SPLITS_V:
        if hybrids:
            compare(q, f"V4_{split}", [f"RX_C3@{w}", *(v for v, _ in hybrids)], refs=K0EXT, split=split)
    winners = []
    for vid, spec in hybrids:
        ok, du, dv = screen(vid, K0EXT, ref_run=f"RX_C3@{w}")
        log(f"V4 {vid}: delta vs K0ext {VAL_UNSEEN} {du}, val {dv} -> {'pass' if ok else 'no'}")
        if ok:
            winners.append({"id": vid, "spec": spec})
    write_json("V4", {"scores": scores, "ref": ref, "picks": picks, "winners": winners})


def phaseV7(q, w):
    """V7, longer training: K0ext and RX_C3 with 30 epochs (full protocol otherwise, seed 1), against the 10-epoch
    runs of phase RT, on val and val_unseen (no test set: informational)."""
    bext = load_bext()
    ext = ext_overrides(bext["ext_noise"])
    k0_run(q, "V7_K0ext30", ["seed=1", "trainer.max_epochs=30", *ext])
    variant(q, "V7_RX_C3_30", w, s5p8(), pc1_ovr=[*ext, "trainer.max_steps=2000"],
            pc2_ovr=[*ext, "trainer.max_epochs=30"], seed=1, spec=bext, splits=SPLITS_V)
    pesq_table(q, "V7", ["K0extfull_s1", f"F_RX_C3@{w}_s1", "V7_K0ext30", "V7_RX_C3_30"], SPLITS_V,
               note="10 epochs (phase RT, seed 1) against 30 epochs (V7, seed 1).")


def phaseVF(q, w):
    """Final confirmation: the winners of V2-V6 with the full protocol (3 seeds), on the test set, val and val_unseen,
    against K0extfull, next to RX_C3 and the V1 blend."""
    bext = load_bext()
    ext = ext_overrides(bext["ext_noise"])
    c3 = [f"F_RX_C3@{w}_s{seed}" for seed in (1, 2, 3)]
    groups = {"K0extfull": K0EXTFULL, "RX_C3": c3}
    v1 = read_json("V1") or {}
    if v1.get("w") is not None:
        groups[f"blend K0extfull + Bext (w {v1['w']:g})"] = [f"V1_K0extfull_s{seed}_w{v1['w']:g}" for seed in (1, 2, 3)]
    jobs = []      # (label, [run per seed], function(seed, run))
    v2 = read_json("V2") or {}
    if v2.get("winner"):
        p = v2["p"]

        def run_v2(seed, vid, p=p):
            variant(q, vid, w, s5p8(), pc1_ovr=[*ext, "trainer.max_steps=2000"],
                    pc2_ovr=[*ext, "trainer.max_epochs=10", f"fusion_dropout={p}"], seed=seed, spec=bext,
                    splits=SPLITS_V, test=True)
            off_evals(q, vid, SPLITS_V, test=True)
        jobs.append((f"RX_C3 + condition dropout {p:g}", "F_" + v2["winner"], run_v2))
    v3 = read_json("V3") or {}
    if v3.get("winner"):
        def run_v3(seed, vid, setting=v3["setting"]):
            k0_run(q, vid, [f"seed={seed}", "trainer.max_epochs=10", *ext_overrides(setting)], test=True)
        jobs.append((f"K0, recipe {v3['winner']}", "F_" + v3["winner"], run_v3))
    for win in (read_json("V4") or {}).get("winners", []):
        def run_v4(seed, vid, spec=win["spec"]):
            variant(q, vid, w, s5p8(), pc1_ovr=[*ext, "trainer.max_steps=2000"],
                    pc2_ovr=[*ext, "trainer.max_epochs=10"], seed=seed, spec=spec, splits=SPLITS_V, test=True)
        jobs.append((f"RX_C3 with {win['spec']['tag']}", "F_" + win["id"], run_v4))
    for win in (read_json("V5") or {}).get("winners", []):
        def run_v5(seed, vid, win=win):
            variant(q, vid, win["arch"], win["arch_ovr"], pc1_ovr=[*ext, "trainer.max_steps=2000"],
                    pc2_ovr=[*ext, "trainer.max_epochs=10", *win["pc2"]], seed=seed, spec=bext, splits=SPLITS_V,
                    test=True)
        jobs.append((win["id"], "F_" + win["id"], run_v5))
    v6 = read_json("V6") or {}
    if v6.get("winner"):
        def run_v6(seed, vid, lam=v6["weight"]):
            teacher = f"{LOGS}/F_RX_C3@{w}_s{seed}_pc2/last.ckpt"
            k0_run(q, vid, [f"seed={seed}", "trainer.max_epochs=10", *ext,
                            "distill=" + _flow({"teacher_checkpoint": teacher, "weight": lam})], test=True, need=14000)
        jobs.append((f"K0 + distillation of RX_C3 (weight {v6['weight']:g})", "F_" + v6["winner"], run_v6))
    if q.dry_run and not jobs:
        log("dry run: no decision files, VF would only report the existing runs")
    for seed in (1, 2, 3):          # seed 1 of every configuration first
        for label, prefix, fn in jobs:
            fn(seed, f"{prefix}_s{seed}")
    for label, prefix, _ in jobs:
        groups[label] = [f"{prefix}_s{seed}" for seed in (1, 2, 3)]
    runs = [r for label, prefix, _ in jobs for r in groups[label]]
    for split in ("test", *SPLITS_V):
        done = [r for r in (*runs, *c3) if os.path.exists(f"{RESULTS}/{r}/{split}/metrics.csv")]
        if done:
            compare(q, f"VF_{split}", done, refs=K0EXTFULL, split=split)
        seed_table(q, f"VF_seeds_{split}", groups, split=split)
    off = [r for r in runs if os.path.exists(f"{RESULTS}/{r}/test_off/summary.json")]
    if off:
        pesq_table(q, "VF_conditions", off, ("test", *SPLITS_V), ("", "_shuffle", "_off"),
                   note="Normal, condition from another file (_shuffle) and condition off (_off).")


# ------------------------------------------------------------------ phase W (plan, section 14)
VAL_LOW = "val_unseen_low"
W_FINAL_SPLITS = ("test", VAL_UNSEEN, VAL_LOW, "val")
W_CMAG = (10.0, 30.0)                  # compressed-magnitude loss: ~0.45x / ~1.3x the gradient norm of the loss
W_MGAN = (0.3, 1.0)                    # metric discriminator: ~9% / ~30% of the gradient norm of the loss
W_KD = (1.0, 2.0)                      # distillation of the ensemble teacher
W_BLEND = (0.15, 0.25, 0.35, 0.5)      # CleanSpecNet weight of the training-free blend
W_ALPHA = (0.3, 0.4, 0.5, 0.6, 0.7)    # RX_C3 weight of the output ensemble RX_C3 + blend
W_EPOCHS = 30                          # final waveform-only models (V7: 30 epochs > 10)
GUARD_STOI, GUARD_SISDR = -0.003, -0.5   # a new loss may not buy PESQ with STOI / SI-SDR (val_unseen)
TTA_GAIN = 0.01                        # val_unseen gain a test-time augmentation needs (it multiplies the cost)
SOUP_TOL = -0.005                      # a soup is kept unless it loses more than this against its members' mean
AUDIO = "W_audio"                      # results/abl/W_audio/<system>/<split>: enhanced audio for output ensembles
_DATA = "/home/fred/DATASETS/VoiceBank-DEMAND-16k"
W_LABELS = {"test": f"{_DATA}/logfiles/log_testset.txt", VAL_UNSEEN: f"{_DATA}/val_unseen/log_val_unseen.txt",
            VAL_LOW: f"{_DATA}/{VAL_LOW}/log_{VAL_LOW}.txt"}
POL_GAP = 0.05                         # largest val_unseen PESQ loss on the polarity-inverted input (invariance)
# polarity fixes (14.3): random sign of the training pair, random sign of every babble talker / noise excerpt
WP_CANDS = {"WP_flip": {"augment_polarity": 0.5}, "WP_sign": {"random_sign": True},
            "WP_both": {"augment_polarity": 0.5, "random_sign": True}}
# decisions assumed by a dry run when a phase has not written its file yet (exercises the longest path)
W_DRY = {"W1": {"alpha": 0.5, "blend_w": 0.25, "tta": "flip", "soup_ok": True},
         "WP": {"winner": "WP_both", "fix": WP_CANDS["WP_both"]},
         "W4": {"recipe_run": "W4_combo", "recipe": {"cmag": W_CMAG[0], "kd": W_KD[0]}},
         "W5": {"spec": "W5_bidir_final", "checkpoint": f"{LOGS}/N_W5_bidir_final/last.ckpt", "blend_w": 0.25}}


def decision(q, name):
    d = read_json(name)
    if d is None and q.dry_run:
        return dict(W_DRY.get(name, {}))
    return d or {}


def eval_args(split, extra=()):
    """evaluate.py arguments for a split: the composite measures on the test set only."""
    args = ["--filelist", f"filelists/{split}.csv"]
    if split != "test":
        args.append("--fast")
    return [*args, *extra]


def w_eval(q, run, ckpt, splits, extra=()):
    for split in splits:
        q.evaluate(f"{run}_{split}", ckpt, f"{RESULTS}/{run}/{split}", eval_args(split, extra))


def make_soup(q, name, members):
    """Weight soup (scripts/make_soup.py) of the last checkpoints of `members` -> logs/abl/<name>/last.ckpt."""
    out = f"{LOGS}/{name}/last.ckpt"
    q.script(f"soup_{name}", ["scripts/make_soup.py", "--ckpts", *(f"{LOGS}/{m}/last.ckpt" for m in members),
                              "--out", out], out, need=0)
    return out


def audio(q, label, ckpt, split, extra=(), tag=""):
    """Metrics and enhanced audio of a system on a split -> results/abl/W_audio/<label>/<split><tag>; returns the
    dir. tag "_inv": the polarity-inverted input (evaluate.py --invert)."""
    out = f"{RESULTS}/{AUDIO}/{label}/{split}{tag}"
    inv = ["--invert"] if tag == "_inv" else []
    q.evaluate(f"audio_{label}_{split}{tag}", ckpt, out, [*eval_args(split, extra), *inv, "--save_audio"])
    return out


def ensemble(q, name, inputs, weights, split, robust=False, tag=""):
    """Weighted average of saved audio (scripts/ensemble_outputs.py) -> results/abl/<name>/<split><tag>[_r];
    robust: PESQ-WB, STOI, SI-SDR and the robust PESQ only."""
    out = f"{RESULTS}/{name}/{split}{tag}" + ("_r" if robust else "")
    cmd = ["scripts/ensemble_outputs.py", "--inputs", *inputs, "--weights", *(f"{x:g}" for x in weights),
           "--filelist", f"filelists/{split}.csv", "--out_dir", out]
    if split != "test" or robust:
        cmd.append("--fast")
    if robust:
        cmd.append("--robust_pesq")
    q.script(f"ens_{name.replace('/', '_')}_{split}{tag}" + ("_r" if robust else ""), cmd, f"{out}/summary.json",
             need=0)
    return out


def best_key(scores, default):
    scores = {k: v for k, v in scores.items() if v is not None}
    return max(scores, key=scores.get) if scores else default


def mean_table(q, name, groups, splits, note=""):
    """Seed-mean PESQ-WB of each group of runs on every split."""
    if q.dry_run:
        return
    lines = ["| system | runs | " + " | ".join(splits) + " |", "|---|---|" + "---|" * len(splits)]
    for label, runs in groups.items():
        cells = []
        for sp in splits:
            m = mean_pesq(runs, sp)
            cells.append("-" if m is None else f"{m:.4f}")
        lines.append(f"| {label} | {len(runs)} | " + " | ".join(cells) + " |")
    with open(f"{RESULTS}/{name}.md", "w") as fh:
        fh.write(f"PESQ-WB, mean over the runs of each group. {note}\n\n" + "\n".join(lines) + "\n")
    log(f"table written to {RESULTS}/{name}.md")


def screen_w(run, refs):
    """screen() plus the guards against a PESQ gain bought with STOI or SI-SDR (val_unseen).
    Returns (passes, d_unseen, d_val, d_stoi, d_sisdr)."""
    ok, du, dv = screen(run, refs)
    s, ref = summary_of(run, VAL_UNSEEN), [summary_of(r, VAL_UNSEEN) for r in refs]
    if du is None or not s or not all(ref):
        return False, du, dv, None, None
    d_stoi = s["stoi"] - statistics.mean(r["stoi"] for r in ref)
    d_sisdr = s["si_sdr"] - statistics.mean(r["si_sdr"] for r in ref)
    return ok and d_stoi >= GUARD_STOI and d_sisdr >= GUARD_SISDR, du, dv, d_stoi, d_sisdr


def pick_w(tag, runs, refs=K0EXT):
    """The passing run with the best val_unseen delta (screen_w), or None."""
    passed = {}
    for name in runs:
        ok, du, dv, ds, dq = screen_w(name, refs)
        fmt = lambda v, f="{:+.4f}": "-" if v is None else f.format(v)  # noqa: E731
        log(f"{tag} {name}: delta {VAL_UNSEEN} {fmt(du)}, val {fmt(dv)}, STOI {fmt(ds)}, SI-SDR {fmt(dq, '{:+.2f}')} "
            f"-> {'pass' if ok else 'no'}")
        if ok:
            passed[name] = du
    return max(passed, key=passed.get) if passed else None


def ensemble_teachers(seed, w, bext, w1):
    """The output ensemble of W1 (RX_C3 + blend of K0extfull with Bext) of one seed, as distillation teachers."""
    alpha, wb = float(w1.get("alpha", 0.5)), float(w1.get("blend_w", 0.25))
    return [{"checkpoint": f"{LOGS}/F_RX_C3@{w}_s{seed}_pc2/last.ckpt", "weight": alpha},
            {"checkpoint": f"{LOGS}/K0extfull_s{seed}/last.ckpt", "blend_spec": bext["checkpoint"], "blend_w": wb,
             "weight": round(1 - alpha, 6)}]


def recipe_overrides(recipe, seed, w=None, bext=None, w1=None):
    """Training overrides of a waveform-branch recipe {cmag: weight, mgan: weight, kd: weight} for one seed."""
    ovr = []
    if recipe.get("cmag"):
        ovr.append("loss.cmag=" + _flow({"weight": float(recipe["cmag"])}))
    if recipe.get("mgan"):
        ovr.append("loss.metric_gan=" + _flow({"weight": float(recipe["mgan"])}))
    if recipe.get("kd"):
        teachers = ensemble_teachers(seed, w, bext, w1)
        ovr.append("distill=" + _flow({"weight": float(recipe["kd"]), "teachers": teachers}))
    return ovr


def phaseW0(q):
    """W0, data: the low-SNR unseen-noise set (the noise types of val_unseen at -5 / 0 dB,
    scripts/make_val_unseen.py)."""
    q.script("make_val_unseen_low", ["scripts/make_val_unseen.py", "--name", VAL_LOW, "--snrs", "-5", "0",
                                     "--seed", "1", "--rescale"], f"filelists/{VAL_LOW}.csv", need=0)


def phaseW1(q, w):
    """W1, training-free, existing checkpoints: weight soups of the K0ext / K0extfull seeds and the output ensemble of
    the 3 K0extfull seeds; test-time augmentation (sign flip / shifts) of K0extfull; the CleanSpecNet weight of the
    blend K0extfull + Bext; the RX_C3 weight of the output ensemble RX_C3 + blend. Selection on val_unseen (val as a
    guard), decisions in results/abl/W1.json."""
    bext = load_bext()
    blend_args = lambda x: ["--blend_spec", bext["checkpoint"], "--blend_w", f"{x:g}"]  # noqa: E731
    soups = {"K0extfull_soup": K0EXTFULL, "K0ext_soup": K0EXT}
    for name, members in soups.items():
        w_eval(q, name, make_soup(q, name, members), SPLITS_V)
    for split in SPLITS_V:
        ens = [audio(q, run, f"{LOGS}/{run}/last.ckpt", split) for run in K0EXTFULL]
        ensemble(q, "K0extfull_ens3", ens, [1, 1, 1], split)
    for mode in ("flip", "shift"):
        for run in K0EXTFULL:
            w_eval(q, f"W1_{run}_tta{mode}", f"{LOGS}/{run}/last.ckpt", SPLITS_V, ["--tta", mode])
    for x in W_BLEND:                     # 0.25 and 0.5 exist from phase V1
        for run in K0EXTFULL:
            for split in SPLITS_V:
                q.evaluate(f"V1_{run}_w{x:g}_{split}", f"{LOGS}/{run}/last.ckpt", f"{RESULTS}/V1_{run}_w{x:g}/{split}",
                           eval_args(split, blend_args(x)))
    blend_scores = {x: mean_pesq([f"V1_{r}_w{x:g}" for r in K0EXTFULL], VAL_UNSEEN) for x in W_BLEND}
    wb = best_key(blend_scores, 0.25)
    for seed in (1, 2, 3):
        for split in SPLITS_V:
            c3 = audio(q, f"F_RX_C3_s{seed}", f"{LOGS}/F_RX_C3@{w}_s{seed}_pc2/last.ckpt", split)
            bl = audio(q, f"blend_K0extfull_s{seed}_w{wb:g}", f"{LOGS}/K0extfull_s{seed}/last.ckpt", split,
                       blend_args(wb))
            for alpha in W_ALPHA:
                ensemble(q, f"W1_ens{alpha:g}_s{seed}", [c3, bl], [alpha, 1 - alpha], split)
    ens_scores = {a: mean_pesq([f"W1_ens{a:g}_s{s}" for s in (1, 2, 3)], VAL_UNSEEN) for a in W_ALPHA}
    alpha = best_key(ens_scores, 0.5)

    groups = {"K0extfull": K0EXTFULL, "K0extfull soup": ["K0extfull_soup"], "K0extfull, output mean of the 3 seeds":
              ["K0extfull_ens3"], **{f"K0extfull + TTA {m}": [f"W1_{r}_tta{m}" for r in K0EXTFULL]
                                     for m in ("flip", "shift")},
              **{f"blend K0extfull + Bext, w {x:g}": [f"V1_{r}_w{x:g}" for r in K0EXTFULL] for x in W_BLEND},
              **{f"RX_C3 x {a:g} + blend (w {wb:g})": [f"W1_ens{a:g}_s{s}" for s in (1, 2, 3)] for a in W_ALPHA},
              "RX_C3": [f"F_RX_C3@{w}_s{s}" for s in (1, 2, 3)], "K0ext (short)": K0EXT, "K0ext soup": ["K0ext_soup"]}
    mean_table(q, "W1", groups, SPLITS_V, note="Training-free variants of existing checkpoints (phase W1).")
    for split in SPLITS_V:
        compare(q, f"W1_{split}", ["K0extfull_soup", "K0extfull_ens3",
                                   *(f"W1_{r}_tta{m}" for m in ("flip", "shift") for r in K0EXTFULL),
                                   *(f"W1_ens{alpha:g}_s{s}" for s in (1, 2, 3))], refs=K0EXTFULL, split=split)
    if q.dry_run:
        return
    soup_gain = {}
    for name, members in soups.items():
        g = [(summary_of(name, sp), mean_pesq(members, sp)) for sp in SPLITS_V]
        if all(s and m is not None for s, m in g):
            soup_gain[name] = {sp: s["pesq_wb"] - m for sp, (s, m) in zip(SPLITS_V, g)}
    soup_ok = bool(soup_gain) and all(g[VAL_UNSEEN] >= SOUP_TOL and g["val"] >= VAL_GUARD for g in soup_gain.values())
    base = {sp: mean_pesq(K0EXTFULL, sp) for sp in SPLITS_V}
    tta_gain = {}
    for mode in ("flip", "shift"):
        m = {sp: mean_pesq([f"W1_{r}_tta{mode}" for r in K0EXTFULL], sp) for sp in SPLITS_V}
        if all(m[sp] is not None and base[sp] is not None for sp in SPLITS_V):
            tta_gain[mode] = {sp: m[sp] - base[sp] for sp in SPLITS_V}
    good = {k: g[VAL_UNSEEN] for k, g in tta_gain.items() if g[VAL_UNSEEN] >= TTA_GAIN and g["val"] >= VAL_GUARD}
    write_json("W1", {"soup_gain": soup_gain, "soup_ok": soup_ok, "tta_gain": tta_gain, "tta": best_key(good, None),
                      "blend_scores": {f"{k:g}": v for k, v in blend_scores.items()}, "blend_w": wb,
                      "ensemble_scores": {f"{k:g}": v for k, v in ens_scores.items()}, "alpha": alpha})


def phaseW2(q):
    """W2, loss of the waveform branch (K0, short protocol, seed 1, external noise) against K0ext: the power-law
    compressed magnitude loss and the metric discriminator, two weights each. The screening rule of phase V plus the
    STOI / SI-SDR guards; the best passing run -> results/abl/W2.json."""
    ext = ext_overrides(load_bext()["ext_noise"])
    todo = {**{f"W2_cmag{c:g}": {"cmag": c} for c in W_CMAG}, **{f"W2_mgan{g:g}": {"mgan": g} for g in W_MGAN}}
    for name, recipe in todo.items():
        k0_run(q, name, ["seed=1", *ext, *recipe_overrides(recipe, 1)], need=8000)
    for split in SPLITS_V:
        compare(q, f"W2_{split}", list(todo), refs=K0EXT, split=split)
    if q.dry_run:
        return
    winner = pick_w("W2", list(todo))
    write_json("W2", {"winner": winner, "recipe": todo.get(winner)})


def phaseW3(q, w):
    """W3, distillation of the output ensemble of W1 (RX_C3 + blend of K0extfull with Bext, seed 1) into the
    waveform-only model (K0, short protocol, seed 1, external noise), two weights, against K0ext; V6 (RX_C3 alone as
    the teacher) for reference. Decision -> results/abl/W3.json."""
    bext = load_bext()
    ext = ext_overrides(bext["ext_noise"])
    w1 = decision(q, "W1")
    todo = {f"W3_kd{lam:g}": {"kd": lam} for lam in W_KD}
    for name, recipe in todo.items():
        k0_run(q, name, ["seed=1", *ext, *recipe_overrides(recipe, 1, w, bext, w1)], need=8000)
    for split in SPLITS_V:
        compare(q, f"W3_{split}", [*todo, "V6_kd1"], refs=K0EXT, split=split)
    if q.dry_run:
        return
    winner = pick_w("W3", list(todo))
    write_json("W3", {"winner": winner, "recipe": todo.get(winner),
                      "teacher": {"alpha": w1.get("alpha", 0.5), "blend_w": w1.get("blend_w", 0.25)}})


def phaseW4(q, w):
    """W4, the recipe of the final waveform-only model: the W2 and W3 winners and, when both exist, their combination
    (short protocol, seed 1), the best passing one by val_unseen; none -> the plain K0ext recipe.
    Decision -> results/abl/W4.json."""
    bext = load_bext()
    ext = ext_overrides(bext["ext_noise"])
    w1, w2, w3 = decision(q, "W1"), decision(q, "W2"), decision(q, "W3")
    if q.dry_run and not w2 and not w3:
        w2 = {"winner": f"W2_cmag{W_CMAG[0]:g}", "recipe": {"cmag": W_CMAG[0]}}
        w3 = {"winner": f"W3_kd{W_KD[0]:g}", "recipe": {"kd": W_KD[0]}}
    cands = {d["winner"]: d["recipe"] for d in (w2, w3) if d.get("winner")}
    if len(cands) == 2:
        combo = {**w2["recipe"], **w3["recipe"]}
        k0_run(q, "W4_combo", ["seed=1", *ext, *recipe_overrides(combo, 1, w, bext, w1)], need=8000)
        cands["W4_combo"] = combo
        for split in SPLITS_V:
            compare(q, f"W4_{split}", list(cands), refs=K0EXT, split=split)
    if q.dry_run:
        return
    best = pick_w("W4", list(cands))
    write_json("W4", {"recipe_run": best, "recipe": cands.get(best) or {}})


def phaseW5(q, w):
    """W5, spectral branch of the offline systems: Bext with the bidirectional backbone (V4_bidir, the only V4
    candidate that beat Bext) retrained for the Bext budget, alone and blended with K0extfull (3 seeds, W_BLEND)
    against the blends with Bext. The best (spectral branch, w) by val_unseen -> results/abl/W5.json."""
    bext = load_bext()
    name = "W5_bidir_final"
    ck = n_run(q, name, [*bext["overrides"], *(bidir_overrides(bext) or []), f"trainer.max_epochs={spec_epochs(bext)}"])
    spec = f"{ck}/last.ckpt" if ck else None
    if ck:
        r_eval(q, name, spec, VAL_UNSEEN)
    r_table(q, "W5_spec", ["R_ext50", "R_ext50_final", "V4_bidir", name])
    runs = {}
    for tag, ckpt, prefix in (("Bext", bext["checkpoint"], "V1_{run}_w{x:g}"), (name, spec, "W5_{run}_bidir_w{x:g}")):
        if ckpt is None:
            continue
        for x in W_BLEND:
            group = []
            for run in K0EXTFULL:
                vid = prefix.format(run=run, x=x)
                for split in SPLITS_V:
                    q.evaluate(f"{vid}_{split}", f"{LOGS}/{run}/last.ckpt", f"{RESULTS}/{vid}/{split}",
                               eval_args(split, ["--blend_spec", ckpt, "--blend_w", f"{x:g}"]))
                group.append(vid)
            runs[(tag, ckpt, x)] = group
    mean_table(q, "W5", {f"K0extfull + {t}, w {x:g}": g for (t, _, x), g in runs.items()}, SPLITS_V,
               note="Training-free blends of K0extfull (3 seeds) with each spectral branch.")
    if q.dry_run:
        return
    scores = {k: mean_pesq(g, VAL_UNSEEN) for k, g in runs.items()}
    tag, ckpt, x = best_key(scores, ("Bext", bext["checkpoint"], 0.25))
    write_json("W5", {"spec": tag, "checkpoint": ckpt, "blend_w": x,
                      "scores": {f"{t} w{x_:g}": v for (t, _, x_), v in scores.items()}})


def fix_overrides(fix, ext_setting):
    """External-noise overrides of K0 runs with a polarity fix {augment_polarity, random_sign, weights} (or none)."""
    fix = fix or {}
    setting = dict(ext_setting)
    if fix.get("random_sign"):
        setting["random_sign"] = True
    if fix.get("weights"):
        setting["weights"] = dict(fix["weights"])
    ovr = ext_overrides(setting)
    if fix.get("augment_polarity"):
        ovr.append(f"augment_polarity={float(fix['augment_polarity']):g}")
    return ovr


def inv_eval(q, run, ckpt, split=VAL_UNSEEN, extra=()):
    """evaluate.py --invert (polarity-inverted input, output inverted back) -> results/abl/<run>/<split>_inv."""
    q.evaluate(f"{run}_{split}_inv", ckpt, f"{RESULTS}/{run}/{split}_inv", [*eval_args(split, extra), "--invert"])


def pol_gap(runs, split=VAL_UNSEEN):
    """Mean PESQ on the normal input minus on the polarity-inverted input, or None."""
    normal, inverted = mean_pesq(runs, split), mean_pesq(runs, f"{split}_inv")
    return None if normal is None or inverted is None else normal - inverted


def phaseWP(q):
    """WP, polarity (plan, 14.3). The models trained with MUSAN babble fail on a polarity-inverted input (K0extfull:
    PESQ 3.19 -> 1.49 on val_unseen), the ones trained without babble do not (V3_noise: 3.21 -> 3.19). Candidates (K0,
    short protocol, seed 1): random sign of the training pair, random sign of every babble talker / noise excerpt,
    both; and V3_noise (MUSAN noise only, 3 seeds). Kept: the best val_unseen delta against K0ext among the
    candidates whose inverted-input loss is <= POL_GAP and whose delta is >= -delta_min (val as a guard); a one-seed
    winner is confirmed with seeds 2 and 3. Decision -> results/abl/WP.json."""
    base = load_bext()["ext_noise"]
    v3 = ["V3_noise", "V3_noise_s2", "V3_noise_s3"]
    for run in (*K0EXT, *v3):
        inv_eval(q, run, f"{LOGS}/{run}/last.ckpt")
    for name, fix in WP_CANDS.items():
        k0_run(q, name, ["seed=1", *fix_overrides(fix, base)], need=8000)
        inv_eval(q, name, f"{LOGS}/{name}/last.ckpt")
    cands = {**{n: ([n], f) for n, f in WP_CANDS.items()},
             "V3_noise": (v3, {"weights": V3_RECIPES["V3_noise"]["weights"]})}
    pesq_table(q, "WP", [*K0EXT, *v3, *WP_CANDS], SPLITS_V, ("", "_inv"),
               note="Normal input and polarity-inverted input (_inv, output inverted back).")
    if q.dry_run:
        return
    bar = dmin_split(K0EXT, VAL_UNSEEN)

    def judge(name, runs):
        d = [mean_delta(r, K0EXT, SPLITS_V)[1] for r in runs]
        gap = pol_gap(runs)
        if not all(d) or gap is None:
            return False, None, gap
        du, dv = statistics.mean(x[1] for x in d), statistics.mean(x[0] for x in d)
        ok = gap <= POL_GAP and du >= -bar and dv >= VAL_GUARD
        log(f"WP {name} ({len(runs)} seed(s)): delta {VAL_UNSEEN} {du:+.4f}, val {dv:+.4f}, inverted-input loss "
            f"{gap:+.4f} -> {'pass' if ok else 'no'}")
        return ok, du, gap

    log(f"WP K0ext: inverted-input loss {pol_gap(K0EXT)}")
    results = {name: judge(name, runs) for name, (runs, _) in cands.items()}
    ok = {n: r[1] for n, r in results.items() if r[0]}
    winner = best_key(ok, None)
    if winner in WP_CANDS:                       # one seed so far: seeds 2 and 3
        seeds = [winner, f"{winner}_s2", f"{winner}_s3"]
        for seed, name in zip((2, 3), seeds[1:]):
            k0_run(q, name, [f"seed={seed}", *fix_overrides(WP_CANDS[winner], base)], need=8000)
            inv_eval(q, name, f"{LOGS}/{name}/last.ckpt")
        cands[winner] = (seeds, WP_CANDS[winner])
        if not judge(winner, seeds)[0]:
            ok.pop(winner)
            winner = "V3_noise" if "V3_noise" in ok else None
    write_json("WP", {"winner": winner, "fix": cands[winner][1] if winner else None,
                      "inverted_loss": {n: r[2] for n, r in results.items()}, "k0ext_inverted_loss": pol_gap(K0EXT)})


def w_final_runs(q, w):
    """Final waveform-only runs: (control seeds, recipe seeds or None, W4 recipe, WP polarity fix). The control is K0
    with 30 epochs and the polarity fix; without a fix it is K0ext30 (V7_K0ext30 is its seed 1)."""
    fix = decision(q, "WP").get("fix") or {}
    control = [f"K0ext30p_s{seed}" for seed in (1, 2, 3)] if fix else ["V7_K0ext30", "K0ext30_s2", "K0ext30_s3"]
    recipe = decision(q, "W4").get("recipe") or {}
    simple = [f"S30_s{seed}" for seed in (1, 2, 3)] if recipe else None
    return control, simple, recipe, fix


def w_best_params(q):
    """(alpha, TTA mode or None, spectral branch checkpoint, blend weight) of the offline systems."""
    bext = load_bext()
    w1, w5 = decision(q, "W1"), decision(q, "W5")
    return (float(w1.get("alpha", 0.5)), w1.get("tta"), w5.get("checkpoint") or bext["checkpoint"],
            float(w5.get("blend_w") or w1.get("blend_w") or 0.25))


def w_systems(q, w, tag=""):
    """Audio of the offline systems and their output ensembles for every final split (tag "_inv": on val_unseen with
    the polarity-inverted input)."""
    bext = load_bext()
    control, simple, _, _ = w_final_runs(q, w)
    wave = simple or control
    alpha, tta, spec, wb = w_best_params(q)
    t_args, t_tag = (["--tta", tta], f"_tta{tta}") if tta else ([], "")
    splits = (VAL_UNSEEN,) if tag else W_FINAL_SPLITS
    for seed in (1, 2, 3):
        c3_ck = f"{LOGS}/F_RX_C3@{w}_s{seed}_pc2/last.ckpt"
        for split in splits:
            bl = audio(q, f"blend_{wave[seed - 1]}{t_tag}", f"{LOGS}/{wave[seed - 1]}/last.ckpt", split,
                       ["--blend_spec", spec, "--blend_w", f"{wb:g}", *t_args], tag)
            c3 = audio(q, f"F_RX_C3_s{seed}{t_tag}", c3_ck, split, t_args, tag)
            ensemble(q, f"WF_best_s{seed}", [c3, bl], [alpha, 1 - alpha], split, tag=tag)
            c3_0 = audio(q, f"F_RX_C3_s{seed}", c3_ck, split, (), tag)
            bl_0 = audio(q, f"blend_K0extfull_s{seed}_w0.25", f"{LOGS}/K0extfull_s{seed}/last.ckpt", split,
                         ["--blend_spec", bext["checkpoint"], "--blend_w", "0.25"], tag)
            ensemble(q, f"WF_refens_s{seed}", [c3_0, bl_0], [0.5, 0.5], split, tag=tag)


def phaseWF(q, w):
    """WF, final runs, 3 seeds, seeds interleaved, on test / val_unseen / val_unseen_low / val:
    simplest model: K0 with 30 epochs and the WP polarity fix (control), the W4 recipe with 30 epochs on top of it
    (S30), and their weight soups; best offline system: the blend of S30 with the W5 spectral branch, and the output
    ensemble RX_C3 x alpha + that blend (alpha, TTA from W1), next to the ensemble of the analysis (RX_C3 + blend
    K0extfull / Bext, equal weights). Everything against K0extfull."""
    bext = load_bext()
    control, simple, recipe, fix = w_final_runs(q, w)
    data = fix_overrides(fix, bext["ext_noise"])
    w1 = decision(q, "W1")
    alpha, tta, spec, wb = w_best_params(q)
    log(f"WF: polarity fix {fix or 'none'}, recipe {recipe or 'K0ext'}, spectral branch {spec} (w {wb:g}), "
        f"RX_C3 weight {alpha:g}, TTA {tta}")
    for name in K0EXTFULL:
        w_eval(q, name, f"{LOGS}/{name}/last.ckpt", W_FINAL_SPLITS)
    for seed in (1, 2, 3):
        name = control[seed - 1]
        k0_run(q, name, [f"seed={seed}", f"trainer.max_epochs={W_EPOCHS}", *data], splits=(), need=8000)
        w_eval(q, name, f"{LOGS}/{name}/last.ckpt", W_FINAL_SPLITS)
        if simple:
            k0_run(q, simple[seed - 1], [f"seed={seed}", f"trainer.max_epochs={W_EPOCHS}", *data,
                                         *recipe_overrides(recipe, seed, w, bext, w1)], splits=(), need=8000)
            w_eval(q, simple[seed - 1], f"{LOGS}/{simple[seed - 1]}/last.ckpt", W_FINAL_SPLITS)
    for name, members in w_soups(q, w).items():
        w_eval(q, name, make_soup(q, name, members), W_FINAL_SPLITS)
    w_systems(q, w)
    groups = w_groups(q, w)
    runs = [r for label, rs in groups.items() if label != "K0extfull (10 epochs)" for r in rs]
    for split in W_FINAL_SPLITS:
        done = [r for r in runs if os.path.exists(f"{RESULTS}/{r}/{split}/metrics.csv")]
        if done:
            compare(q, f"WF_{split}", done, refs=K0EXTFULL, split=split)
        seed_table(q, f"WF_seeds_{split}", groups, split=split)


def w_soups(q, w):
    control, simple, _, fix = w_final_runs(q, w)
    soups = {"K0extfull_soup": K0EXTFULL, ("K0ext30p_soup" if fix else "K0ext30_soup"): control}
    if simple:
        soups["S30_soup"] = simple
    return soups


def w_groups(q, w):
    """The systems of the final comparison: label -> result runs (relative to results/abl), one per seed."""
    control, simple, recipe, fix = w_final_runs(q, w)
    wave = simple or control
    _, tta, _, _ = w_best_params(q)
    t_tag = f"_tta{tta}" if tta else ""
    soups = list(w_soups(q, w))
    pol = " + polarity fix" if fix else ""
    groups = {"K0extfull (10 epochs)": K0EXTFULL, "K0extfull soup": ["K0extfull_soup"],
              f"K0ext 30 epochs{pol}": control, f"K0ext 30 epochs{pol}, soup": [soups[1]]}
    if fix:
        groups["K0ext 30 epochs, no polarity fix (seed 1)"] = ["V7_K0ext30"]
    if simple:
        groups[f"S30 ({', '.join(f'{k} {v:g}' for k, v in recipe.items() if v)}{pol}, 30 epochs)"] = simple
        groups["S30 soup"] = ["S30_soup"]
    groups[f"blend {'S30' if simple else 'K0ext30'} + spectral branch{t_tag}"] = \
        [f"{AUDIO}/blend_{x}{t_tag}" for x in wave]
    groups["RX_C3"] = [f"{AUDIO}/F_RX_C3_s{s}" for s in (1, 2, 3)]
    groups["ensemble RX_C3 + blend (best)"] = [f"WF_best_s{s}" for s in (1, 2, 3)]
    groups["ensemble RX_C3 + blend K0extfull (analysis 12.11)"] = [f"WF_refens_s{s}" for s in (1, 2, 3)]
    return groups


def phaseWR(q, w):
    """WR, robustness report of the final systems (scripts/robustness_report.py -> results/abl/W_robustness.md):
    seed spread, robust PESQ on the test set, share of files more than 0.1 below K0extfull, worst noise type,
    low-SNR set, polarity-inverted input (val_unseen) and streaming inference of the waveform-only models."""
    control, simple, _, _ = w_final_runs(q, w)
    alpha, tta, _, _ = w_best_params(q)
    t_tag = f"_tta{tta}" if tta else ""
    wave = simple or control
    groups = w_groups(q, w)
    for label, runs in groups.items():
        for r in runs:
            if r.startswith(f"{AUDIO}/"):
                ensemble(q, r, [f"{RESULTS}/{r}/test"], [1], "test", robust=True)
            elif not r.startswith("WF_"):
                ck = f"{LOGS}/{r}/last.ckpt"
                q.evaluate(f"{r}_test_r", ck, f"{RESULTS}/{r}/test_r",
                           ["--filelist", "filelists/test.csv", "--fast", "--robust_pesq"])
                inv_eval(q, r, ck)
    w_systems(q, w, tag="_inv")
    for seed in (1, 2, 3):
        ensemble(q, f"WF_best_s{seed}", [f"{RESULTS}/{AUDIO}/F_RX_C3_s{seed}{t_tag}/test",
                                         f"{RESULTS}/{AUDIO}/blend_{wave[seed - 1]}{t_tag}/test"],
                 [alpha, 1 - alpha], "test", robust=True)
        ensemble(q, f"WF_refens_s{seed}", [f"{RESULTS}/{AUDIO}/F_RX_C3_s{seed}/test",
                                           f"{RESULTS}/{AUDIO}/blend_K0extfull_s{seed}_w0.25/test"],
                 [0.5, 0.5], "test", robust=True)
    soups = list(w_soups(q, w))
    stream = {}
    for r in ("K0extfull_s1", control[0], soups[1], *((simple[0], "S30_soup") if simple else ())):
        q.evaluate(f"{r}_test_stream", f"{LOGS}/{r}/last.ckpt", f"{RESULTS}/{r}/test_stream",
                   ["--filelist", "filelists/test.csv", "--fast", "--streaming"])
        stream[r] = {"streaming": f"{RESULTS}/{r}/test_stream", "offline": f"{RESULTS}/{r}/test"}
    spec = {"ref": "K0extfull (10 epochs)",
            "splits": {"test": {"labels": W_LABELS["test"], "robust": "test_r"},
                       VAL_UNSEEN: {"labels": W_LABELS[VAL_UNSEEN], "robust": None, "inverted": f"{VAL_UNSEEN}_inv"},
                       VAL_LOW: {"labels": W_LABELS[VAL_LOW], "robust": None}},
            "systems": {label: [f"{RESULTS}/{r}" for r in runs] for label, runs in groups.items()},
            "streaming": stream}
    if q.dry_run:
        return
    with open(f"{RESULTS}/W_robustness_spec.json", "w") as fh:
        json.dump(spec, fh, indent=2)
    q._run([PY, "scripts/robustness_report.py", "--spec", f"{RESULTS}/W_robustness_spec.json",
            "--out", f"{RESULTS}/W_robustness.md"], f"{LOGS}/W_robustness.log")
    log(f"robustness report written to {RESULTS}/W_robustness.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phases", nargs="+", required=True,
                    choices=["0", "1", "N", "Nd", "Nx", "2", "3", "4", "C", "5", "6", "U", "R", "RT",
                             "V1", "V2", "V3", "V4", "V5", "V6", "V7", "VF",
                             "W0", "W1", "W2", "W3", "W4", "WP", "W5", "WF", "WR"])
    ap.add_argument("--winner", help="phase-1 architecture (configs/model/fusion/<name>.yaml), or 'auto'")
    ap.add_argument("--with_nc", action="store_true", help="track N: also run the scale variants (N-c)")
    ap.add_argument("--signal", help="phase-2 winning signal (fusion.signal value)")
    ap.add_argument("--operator", help="phase-3 winning operator")
    ap.add_argument("--finalists", nargs="*", default=[],
                    help="variant ids for phase 6, or 'auto' (the winner + the best offline configuration)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)
    # check the arguments of every phase before the first one starts
    for ph in args.phases:
        if (ph in ("Nd", "Nx", "2", "3", "4", "C", "5", "R", "RT", "V1", "V2", "V4", "V5", "V6", "V7", "VF",
                   "W1", "W3", "W4", "W5", "WF", "WR") or (ph == "6" and args.finalists == ["auto"])) \
                and not args.winner:
            raise SystemExit(f"phase {ph} needs --winner")
        if ph == "6" and not args.finalists:
            raise SystemExit("phase 6 needs --finalists (variant ids, or auto)")
    q = Queue(args.dry_run)
    for ph in args.phases:
        log(f"===== phase {ph}")
        if ph == "0":
            phase0(q)
        elif ph == "1":
            phase1(q)
        elif ph == "N":
            phaseN(q, args.with_nc)
        elif ph == "Nd":
            phaseNd(q, resolve_winner(args.winner))
        elif ph == "Nx":
            phaseNx(q, resolve_winner(args.winner))
        elif ph == "2":
            phase2(q, resolve_winner(args.winner))
        elif ph == "3":
            phase3(q, resolve_winner(args.winner), args.signal)
        elif ph == "4":
            phase4(q, resolve_winner(args.winner), args.signal, args.operator)
        elif ph == "C":
            phaseC(q, resolve_winner(args.winner))
        elif ph == "5":
            phase5(q, resolve_winner(args.winner), args.signal, args.operator)
        elif ph == "6":
            phase6(q, resolve_finalists(q, args.finalists, args.winner))
        elif ph == "U":
            phaseU(q)
        elif ph == "R":
            phaseR(q, resolve_winner(args.winner))
        elif ph == "RT":
            phaseRT(q, resolve_winner(args.winner))
        elif ph == "V3":
            phaseV3(q)
        elif ph == "W0":
            phaseW0(q)
        elif ph == "W2":
            phaseW2(q)
        elif ph == "WP":
            phaseWP(q)
        elif ph.startswith("W"):
            {"W1": phaseW1, "W3": phaseW3, "W4": phaseW4, "W5": phaseW5, "WF": phaseWF, "WR": phaseWR}[ph](
                q, resolve_winner(args.winner))
        elif ph.startswith("V"):
            {"V1": phaseV1, "V2": phaseV2, "V4": phaseV4, "V5": phaseV5, "V6": phaseV6, "V7": phaseV7,
             "VF": phaseVF}[ph](q, resolve_winner(args.winner))
    log("queue done")


if __name__ == "__main__":
    main()
