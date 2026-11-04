"""evaluation.metrics.map_surviving_crashes: a worker that dies (PESQ can segfault) must not block the evaluation."""

import os
import signal

import numpy as np
import pytest

from evaluation.metrics import evaluate_many, map_surviving_crashes, summarize


def square_or_crash(x):
    if x in (3, 11):
        os.kill(os.getpid(), signal.SIGSEGV)
    return x * x


def test_crashing_jobs_get_an_error_and_the_others_their_result():
    out = map_surviving_crashes(square_or_crash, list(range(20)), num_workers=4)
    assert out[3] == out[11] == {"error": "worker process crashed"}
    assert [r for i, r in enumerate(out) if i not in (3, 11)] == [i * i for i in range(20) if i not in (3, 11)]
    assert map_surviving_crashes(square_or_crash, [], num_workers=4) == []


def test_evaluate_many_in_parallel_matches_serial():
    rng = np.random.default_rng(0)
    t = np.arange(16000) / 16000
    clean = (0.3 * np.sin(2 * np.pi * 220 * t) * (1 + np.sin(2 * np.pi * 3 * t))).astype(np.float32)
    pairs = [(clean, clean + rng.normal(0, s, clean.shape).astype(np.float32)) for s in (0.01, 0.03, 0.1)]
    serial = evaluate_many(pairs, full=False, num_workers=1)
    parallel = evaluate_many(pairs, full=False, num_workers=3)
    assert summarize(parallel) == pytest.approx(summarize(serial), rel=1e-9)
