#!/usr/bin/env python3
"""Synthetic sessions for exercising shared_maneuver_test.py.

Nothing here is real data. Every parameter of the generator has one of four
sources, recorded next to it in ``SimParams``:

  paper       taken from Gulati et al. 2026 (section/figure given)
  literature  taken from published physiology
  fitted      fitted by fit_to_paper.py so the simulated sessions reproduce the
              correlation statistics the paper reports (Tables 1-2, Fig. 4)
  assumed     not reported anywhere; varied in sensitivity checks

Each simulated session has
  * a brain-layer dHbO trace = the session's physiological brain signal
    (task response + spontaneous fluctuations) + inversion/measurement noise;
  * a scalp-layer dHbO trace = its own task response + its own fluctuations;
  * a Temple-BF trace, lagging by ``bf_lag_s``:
        (1 - w) * the average brain response to the maneuver
          + w   * this session's physiological brain signal
          + the device's own noise.
    w = 0 is a device that only follows the maneuver ("tilt");
    w = 1 is a device that follows the session's brain physiology ("person").
    Equivalently, w is the share of the session's deviation from the average
    brain response that Temple-BF tracks.

Usage
-----
    python simulate.py demo                 # one simulated study per device
    python simulate.py write OUT --mix 1    # OUT/manifest.csv + session CSVs
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

from shared_maneuver_test import Session, fmt_p, lowpass1, run

FIT_FILE = Path(__file__).with_name("paper_fit.json")
FAST_TAU_S = 2.0  # time constant of the fast (hydrostatic) brain component


@dataclass(frozen=True)
class SimParams:
    # paper, Sec. 2.3 / Fig. 2: five blocks, task in blocks 2 and 4
    block_s: float = 180.0
    # paper, Sec. 3.1: brain dHbO reaches half its block peak in 10-20 s, so a
    # first-order response has tau = t_half / ln 2 = 14.4-28.9 s (peak at
    # ~3-4 tau = 60-90 s, also as reported). Drawn per session.
    tau_brain_s: tuple[float, float] = (10 / math.log(2), 20 / math.log(2))
    # literature: spontaneous cerebral haemodynamic oscillations lie in the
    # very-low-frequency (~0.01-0.05 Hz) and low-frequency/Mayer-wave
    # (~0.1 Hz) bands (Obrig et al. 2000, NeuroImage 12:623)
    band_hz: tuple[float, float] = (0.01, 0.12)
    # assumed: transition gaps between blocks are not reported
    gap_s: tuple[float, float] = (8.0, 20.0)
    # assumed: share of the brain trace's session-specific variance that is
    # physiology (trackable by a device) rather than inversion noise
    phys_share: float = 0.5
    # assumed: small white noise on every trace (before the paper's smoothing)
    white_noise: float = 0.05
    # fitted: SD of session-specific brain variation, in units of the task response
    brain_noise: float = 0.3
    # fitted: SD of Temple-BF's own noise, in units of the task response
    bf_noise: float = 0.3
    # fitted (to the paper's 13-15 s median optimal lag, Sec. 3.2): delay of
    # Temple-BF behind the brain, s
    bf_lag_s: float = 14.0
    # fitted (to the Fig. 4 group-mean r, which is below 1, so the average
    # Temple-BF and brain shapes differ): a systematic drift in Temple-BF from
    # the first to the last event, in task-response units. A stand-in for any
    # shape difference; it is shared by every session, so it lowers real- and
    # mixed-pair r alike and cannot create or hide a gap.
    bf_drift: float = 0.0
    # optional, fitted when enabled (head-down tilt): Temple-BF's own response
    # smoothing, first-order time constant in s (0 = none). With smoothing,
    # part of the 13-15 s lag is the device's sluggishness, not pure delay.
    bf_tau_s: float = 0.0
    # optional, fitted when enabled (head-down tilt): share of the brain
    # response that follows the maneuver within ~2 s (a hydrostatic shift),
    # the rest rising with tau_brain_s. 0 = single slow response.
    brain_fast_share: float = 0.0
    # optional, fitted when enabled (head-down tilt): partial adaptation of
    # the brain response - after its peak it declines by this share of the
    # step, with time constant brain_adapt_tau_s. Motivated by Sec. 3.1, which
    # places the peak at 60-90 s inside 180 s blocks rather than at the end.
    brain_adapt: float = 0.0
    brain_adapt_tau_s: float = 60.0
    # optional, fitted when enabled (head-down tilt): a short transient in
    # Temple-BF at every block transition (same sign at onset and offset, as a
    # movement-related artefact would be), in task-response units.
    bf_transient: float = 0.0
    # fitted: scalp task-response amplitude (signed) relative to its own noise SD
    scalp_amp: float = 1.0
    # fitted: scalp response time constant, s
    scalp_tau_s: float = 8.0


DEFAULT_BLOCKS = {"hdt": 180.0, "squat": 90.0, "supine": 180.0}


def default_params() -> dict[str, SimParams]:
    return {p: SimParams(block_s=b) for p, b in DEFAULT_BLOCKS.items()}


@lru_cache(maxsize=None)
def _bandpass(band_hz: tuple[float, float]):
    return signal.butter(2, band_hz, btype="bandpass", fs=1.0, output="sos")


def band_noise(n: int, rng: np.random.Generator, band_hz: tuple[float, float]) -> np.ndarray:
    """Unit-SD Gaussian noise restricted to band_hz (1 Hz samples)."""
    pad = 300
    x = signal.sosfiltfilt(_bandpass(tuple(band_hz)), rng.standard_normal(n + 2 * pad))[pad:-pad]
    return x / x.std()


def task_indicator(t: np.ndarray, events: np.ndarray) -> np.ndarray:
    """0 in baseline blocks (1, 3, 5), 1 in task blocks (2, 4), ramps in gaps."""
    levels = [0.0, 1.0, 0.0, 1.0, 0.0]
    knots_t, knots_v = [], []
    for k, level in enumerate(levels):
        knots_t += [events[2 * k], events[2 * k + 1]]
        knots_v += [level, level]
    return np.interp(t, knots_t, knots_v)


def delay(x: np.ndarray, lag: float) -> np.ndarray:
    """Delay x by lag seconds (fractional, linear interpolation, 1 Hz samples)."""
    t = np.arange(len(x), dtype=float)
    return np.interp(t - lag, t, x)


def draw_session(p: SimParams, rng: np.random.Generator) -> dict:
    """All random draws for one session. They depend only on the structural
    parameters (block length, gaps, brain kinetics range, noise band), not on
    the fitted amplitudes, so a fit can reuse them (common random numbers)."""
    lead_in = rng.uniform(30, 90)  # recording starts before the first block
    events, t0 = [], lead_in
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


def assemble_session(protocol: str, idx: int, p: SimParams, mix: float, d: dict, individuality: float = 1.0) -> Session:
    task = d["task"]
    # Brain. individuality=0: everyone has the average response and no
    # session-specific physiology (inversion noise remains).
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

    # Scalp: its own response and fluctuations.
    scalp_hbo = p.scalp_amp * lowpass1(task, p.scalp_tau_s) + d["scalp"] + p.white_noise * d["white"][1]

    # Temple-BF: the average brain response plus a share `mix` of this
    # session's deviation from it, delayed; a shared drift; its own noise.
    core = (1 - mix) * response(tau_mean) + mix * brain_phys
    if p.bf_tau_s > 0:
        core = lowpass1(core, p.bf_tau_s)
    ev = d["events"]
    ramp = np.clip((d["t"] - ev[0]) / (ev[-1] - ev[0]), 0.0, 1.0)
    temple_bf = (delay(core, p.bf_lag_s) + p.bf_drift * ramp + p.bf_noise * d["bf"]
                 + p.white_noise * d["white"][2])
    if p.bf_transient:
        moving = np.abs(np.gradient(task))  # nonzero while the posture changes
        bump = lowpass1(lowpass1(moving, 5.0), 5.0)
        temple_bf = temple_bf + p.bf_transient * bump / bump.max()

    return Session(
        session_id=f"{protocol}_{idx:02d}",
        subject_id=f"subj_{protocol}_{idx:02d}",
        protocol=protocol,
        events=d["events"],
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
    """Parameters fitted to the paper (fit_to_paper.py) for the given device mix."""
    data = json.loads(path.read_text())
    out = {}
    for fit in data["fits"]:
        if math.isclose(fit["mix"], mix):
            d = fit["params"]
            out[fit["protocol"]] = SimParams(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})
    if not out:
        raise KeyError(f"no fit for mix={mix} in {path}; run fit_to_paper.py")
    return out


def write_dataset(sessions: list[Session], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "sessions").mkdir(exist_ok=True)
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
    layers = ["brain_hbo", "scalp_hbo"]
    kw = dict(n_perm=perms, seed=seed, residual=True)
    print("One simulated 20-session study per device, parameters fitted to the paper")
    print("(lag-adjusted brain-layer r; gap and p(gap) on full sessions; res = residual analysis;")
    print(" see fit_to_paper.py for ranges over many studies)")
    print(f"{'device':<8}{'protocol':<9}{'real':>7}{'mixed':>7}{'gap':>7}{'p(gap)':>8}"
          f"{'ident':>7}{'res real':>9}{'res mix':>8}")
    for name, mix, ind in (("tilt", 0.0, 1.0), ("person", 1.0, 1.0), ("person*", 1.0, 0.0)):
        res = run(simulate_dataset(mix, load_fitted_params(mix), seed=seed, individuality=ind),
                  "temple_bf", layers, **kw)
        for protocol, r in res.items():
            s = r["layers"][("full", "brain_hbo", "lag")]
            q = r["layers"][("residual", "brain_hbo", "lag")]
            print(f"{name:<8}{protocol:<9}{s['median_real']:>7.3f}{s['median_mixed']:>7.3f}"
                  f"{s['median_gap']:>7.3f}{fmt_p(s['p_gap_wilcoxon']):>8}"
                  f"{s['identified']:>4}/{s['n']}{q['median_real']:>9.3f}{q['median_mixed']:>8.3f}")
    print("person* = the 'person' device in a world where every brain responds identically")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Synthetic data for shared_maneuver_test.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="one simulated study per device")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--perms", type=int, default=2000)
    w = sub.add_parser("write", help="write a synthetic dataset to disk")
    w.add_argument("out", type=Path)
    w.add_argument("--mix", type=float, default=1.0, help="0 = tilt-only device, 1 = person-tracking device")
    w.add_argument("--sessions", type=int, default=20, help="sessions per protocol")
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
