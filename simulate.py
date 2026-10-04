#!/usr/bin/env python3
"""Synthetic sessions for mixed_pairs.py.

Temple-BF is modelled as the average brain response plus a share `mix` of the
session's own deviation from it: mix=0 only follows the maneuver, mix=1
follows the session's brain physiology.

    python simulate.py demo
    python simulate.py write OUT --mix 1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import signal

from mixed_pairs import Session, fmt_p, lowpass1, run

FIT_FILE = Path(__file__).with_name("fit.json")
FAST_TAU_S = 2.0


@dataclass(frozen=True)
class SimParams:
    # Fixed from the paper or literature.
    block_s: float = 180.0  # Fig. 2
    tau_brain_s: tuple[float, float] = (10 / math.log(2), 20 / math.log(2))  # Sec. 3.1: half-rise in 10-20 s
    band_hz: tuple[float, float] = (0.01, 0.12)  # spontaneous oscillation bands, Obrig et al. 2000

    # Not reported anywhere; varied in the sensitivity runs.
    gap_s: tuple[float, float] = (8.0, 20.0)
    phys_share: float = 0.5  # trackable (physiological) share of brain variation
    white_noise: float = 0.05

    # Fitted to the paper by fit.py. Amplitudes are in units of the task response.
    brain_noise: float = 0.3
    bf_noise: float = 0.3
    bf_lag_s: float = 14.0
    bf_drift: float = 0.0  # shared shape difference, needed because Fig. 4 group r < 1
    scalp_amp: float = 1.0
    scalp_tau_s: float = 8.0

    # Head-down tilt only (see README).
    bf_tau_s: float = 0.0
    brain_fast_share: float = 0.0
    brain_adapt: float = 0.0
    brain_adapt_tau_s: float = 60.0
    bf_transient: float = 0.0


BLOCK_S = {"hdt": 180.0, "squat": 90.0, "supine": 180.0}


def default_params() -> dict[str, SimParams]:
    return {p: SimParams(block_s=b) for p, b in BLOCK_S.items()}


@lru_cache(maxsize=None)
def _bandpass(band_hz: tuple[float, float]):
    return signal.butter(2, band_hz, btype="bandpass", fs=1.0, output="sos")


def band_noise(n: int, rng: np.random.Generator, band_hz: tuple[float, float]) -> np.ndarray:
    pad = 300
    x = signal.sosfiltfilt(_bandpass(tuple(band_hz)), rng.standard_normal(n + 2 * pad))[pad:-pad]
    return x / x.std()


def task_indicator(t: np.ndarray, events: np.ndarray) -> np.ndarray:
    levels = [0.0, 1.0, 0.0, 1.0, 0.0]
    knots_t, knots_v = [], []
    for k, level in enumerate(levels):
        knots_t += [events[2 * k], events[2 * k + 1]]
        knots_v += [level, level]
    return np.interp(t, knots_t, knots_v)


def delay(x: np.ndarray, lag: float) -> np.ndarray:
    t = np.arange(len(x), dtype=float)
    return np.interp(t - lag, t, x)


def draw_session(p: SimParams, rng: np.random.Generator) -> dict:
    # Only structural parameters are used here, so a fit can reuse the same
    # draws while it changes the amplitudes.
    events, t0 = [], rng.uniform(30, 90)
    for k in range(5):
        events += [t0, t0 + p.block_s]
        t0 += p.block_s + (rng.uniform(*p.gap_s) if k < 4 else 0)
    events = np.array(events)
    n = int(events[-1] + 60)
    t = np.arange(n, dtype=float)
    return {
        "events": events,
        "t": t,
        "task": task_indicator(t, events),
        "tau_brain": rng.uniform(*p.tau_brain_s),
        "phys": band_noise(n, rng, p.band_hz),
        "inversion": band_noise(n, rng, p.band_hz),
        "scalp": band_noise(n, rng, p.band_hz),
        "bf": band_noise(n, rng, p.band_hz),
        "white": rng.standard_normal((3, n)),
    }


def assemble_session(
    protocol: str, idx: int, p: SimParams, mix: float, d: dict, individuality: float = 1.0
) -> Session:
    task = d["task"]
    tau_mean = float(np.mean(p.tau_brain_s))
    tau_b = tau_mean + individuality * (d["tau_brain"] - tau_mean)
    fast = p.brain_fast_share * lowpass1(task, FAST_TAU_S) if p.brain_fast_share else 0.0

    def response(tau):
        r = fast + (1 - p.brain_fast_share) * lowpass1(task, tau)
        if p.brain_adapt:
            r = r - p.brain_adapt * lowpass1(r, p.brain_adapt_tau_s)
        return r

    brain_phys = response(tau_b) + individuality * p.brain_noise * math.sqrt(p.phys_share) * d["phys"]
    inversion = p.brain_noise * math.sqrt(1 - p.phys_share) * d["inversion"]
    brain_hbo = brain_phys + inversion + p.white_noise * d["white"][0]

    scalp_hbo = p.scalp_amp * lowpass1(task, p.scalp_tau_s) + d["scalp"] + p.white_noise * d["white"][1]

    core = (1 - mix) * response(tau_mean) + mix * brain_phys
    if p.bf_tau_s > 0:
        core = lowpass1(core, p.bf_tau_s)
    ev = d["events"]
    ramp = np.clip((d["t"] - ev[0]) / (ev[-1] - ev[0]), 0.0, 1.0)
    temple_bf = delay(core, p.bf_lag_s) + p.bf_drift * ramp + p.bf_noise * d["bf"] + p.white_noise * d["white"][2]
    if p.bf_transient:
        bump = lowpass1(lowpass1(np.abs(np.gradient(task)), 5.0), 5.0)
        temple_bf = temple_bf + p.bf_transient * bump / bump.max()

    return Session(
        session_id=f"{protocol}_{idx:02d}",
        subject_id=f"subj_{protocol}_{idx:02d}",
        protocol=protocol,
        events=ev,
        time=d["t"],
        signals={"temple_bf": temple_bf, "brain_hbo": brain_hbo, "scalp_hbo": scalp_hbo},
    )


def _structure(p: SimParams) -> tuple:
    return (p.block_s, tuple(p.tau_brain_s), tuple(p.band_hz), tuple(p.gap_s))


@lru_cache(maxsize=64)
def _draws(seed: int, n_sessions: int, structures: tuple) -> tuple:
    rng = np.random.default_rng(seed)
    out = []
    for block_s, tau_brain_s, band_hz, gap_s in structures:
        p = SimParams(block_s=block_s, tau_brain_s=tau_brain_s, band_hz=band_hz, gap_s=gap_s)
        out.append(tuple(draw_session(p, rng) for _ in range(n_sessions)))
    return tuple(out)


def simulate_dataset(
    mix: float,
    params: dict[str, SimParams] | None = None,
    n_sessions: int = 20,
    seed: int = 0,
    protocols: tuple[str, ...] | None = None,
    individuality: float = 1.0,
) -> list[Session]:
    params = params or default_params()
    protocols = tuple(protocols or params)
    draws = _draws(seed, n_sessions, tuple(_structure(params[pr]) for pr in protocols))
    return [
        assemble_session(pr, i, params[pr], mix, d, individuality)
        for pr, group in zip(protocols, draws)
        for i, d in enumerate(group)
    ]


def load_fitted_params(mix: float, path: Path = FIT_FILE) -> dict[str, SimParams]:
    out = {}
    for fit in json.loads(path.read_text())["fits"]:
        if math.isclose(fit["mix"], mix):
            d = fit["params"]
            out[fit["protocol"]] = SimParams(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})
    if not out:
        raise KeyError(f"no fit for mix={mix} in {path}; run fit.py")
    return out


def write_dataset(sessions: list[Session], out: Path) -> None:
    (out / "sessions").mkdir(parents=True, exist_ok=True)
    with open(out / "manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["session_id", "subject_id", "protocol", "file", "events"])
        for s in sessions:
            rel = f"sessions/{s.session_id}.csv"
            w.writerow([s.session_id, s.subject_id, s.protocol, rel, ";".join(f"{e:.1f}" for e in s.events)])
            with open(out / rel, "w", newline="") as f2:
                w2 = csv.writer(f2)
                cols = list(s.signals)
                w2.writerow(["time_s", *cols])
                for k, tk in enumerate(s.time):
                    w2.writerow([f"{tk:.1f}", *(f"{s.signals[c][k]:.5f}" for c in cols)])


def demo(seed: int, perms: int) -> None:
    print("One simulated 20-session study per device (lag-adjusted brain-layer r)")
    print(f"{'device':<8}{'protocol':<9}{'real':>7}{'mixed':>7}{'gap':>7}{'p(gap)':>8}"
          f"{'ident':>7}{'res real':>9}{'res mix':>8}")
    for name, mix, ind in (("tilt", 0.0, 1.0), ("person", 1.0, 1.0), ("person*", 1.0, 0.0)):
        res = run(simulate_dataset(mix, load_fitted_params(mix), seed=seed, individuality=ind),
                  "temple_bf", ["brain_hbo", "scalp_hbo"], n_perm=perms, seed=seed)
        for protocol, r in res.items():
            s = r["layers"][("full", "brain_hbo", "lag")]
            q = r["layers"][("residual", "brain_hbo", "lag")]
            print(f"{name:<8}{protocol:<9}{s['median_real']:>7.3f}{s['median_mixed']:>7.3f}"
                  f"{s['median_gap']:>7.3f}{fmt_p(s['p_gap_wilcoxon']):>8}"
                  f"{s['identified']:>4}/{s['n']}{q['median_real']:>9.3f}{q['median_mixed']:>8.3f}")
    print("person* = person-tracking device when every brain responds identically")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Synthetic data for mixed_pairs.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--perms", type=int, default=2000)
    w = sub.add_parser("write")
    w.add_argument("out", type=Path)
    w.add_argument("--mix", type=float, default=1.0)
    w.add_argument("--sessions", type=int, default=20, help="per protocol")
    w.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.cmd == "demo":
        demo(args.seed, args.perms)
    else:
        try:
            params = load_fitted_params(args.mix)
        except (FileNotFoundError, KeyError):
            params = default_params()
        write_dataset(simulate_dataset(args.mix, params, args.sessions, args.seed), args.out)
        print(f"wrote {args.out}/manifest.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
