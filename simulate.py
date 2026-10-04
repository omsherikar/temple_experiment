#!/usr/bin/env python3
"""Synthetic sessions for exercising shared_maneuver_test.py.

Nothing here is real data. Block timings follow the paper (five blocks of
180 s for head-down tilt and stand-to-supine, 90 s for stand-to-squat, with
short transition gaps); every amplitude, time constant and noise level is an
assumption chosen only to make the headline correlations look like the
paper's (r around 0.85-0.9).

Each simulated session has
  * a true brain dHbO trace: the task response (first-order lag with a
    subject-specific time constant and amplitude) plus slow spontaneous
    fluctuations that belong to that session only;
  * a scalp dHbO trace: a smaller, faster, protocol-specific response plus
    its own slow fluctuations and drift;
  * a Temple-BF trace from one of two imaginary devices, lagging the brain
    by about 13 s (the paper's median optimal lag):
      - "tilt":   follows a generic response to the imposed maneuver and
                  never sees the session's own brain fluctuations;
      - "person": follows that session's true brain trace.
    ``--mix w`` blends them: w=0 is "tilt", w=1 is "person".

Usage
-----
    python simulate.py demo                 # print the comparison tables
    python simulate.py write OUT --device person [--mix 0.5]
        # writes OUT/manifest.csv and one CSV per session, the same format
        # shared_maneuver_test.py reads
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from shared_maneuver_test import Session, fmt_p, run

PROTOCOLS = {
    # name: (block duration s, transition gap range s, scalp response sign)
    "hdt": (180, (12, 25), -0.3),
    "squat": (90, (5, 12), 0.6),
    "supine": (180, (10, 20), 0.8),
}
BF_LAG_S = 13


def first_order(x: np.ndarray, tau: float) -> np.ndarray:
    a = 1.0 / tau
    y = np.zeros_like(x)
    for i in range(1, len(x)):
        y[i] = y[i - 1] + a * (x[i - 1] - y[i - 1])
    return y


def slow_noise(n: int, rng: np.random.Generator, tau: float = 25.0) -> np.ndarray:
    """Unit-SD AR(1) noise with a ~tau s correlation time (1 Hz samples)."""
    phi = np.exp(-1.0 / tau)
    e = rng.standard_normal(n + 200) * np.sqrt(1 - phi**2)
    x = np.zeros_like(e)
    for i in range(1, len(e)):
        x[i] = phi * x[i - 1] + e[i]
    return x[200:]


def unit(x: np.ndarray) -> np.ndarray:
    return x / (x.std() or 1.0)


def task_indicator(t: np.ndarray, events: np.ndarray) -> np.ndarray:
    """0 in baseline blocks (1, 3, 5), 1 in task blocks (2, 4), ramps in gaps."""
    levels = [0.0, 1.0, 0.0, 1.0, 0.0]
    knots_t, knots_v = [], []
    for k, level in enumerate(levels):
        knots_t += [events[2 * k], events[2 * k + 1]]
        knots_v += [level, level]
    return np.interp(t, knots_t, knots_v)


def simulate_session(
    protocol: str, idx: int, mix: float, rng: np.random.Generator, individuality: float = 1.0
) -> Session:
    block, (gmin, gmax), scalp_sign = PROTOCOLS[protocol]
    lead_in = rng.uniform(30, 90)  # recording starts before the first block
    events, t0 = [], lead_in
    for k in range(5):
        events += [t0, t0 + block]
        t0 += block + (rng.uniform(gmin, gmax) if k < 4 else 0)
    events = np.array(events)
    n = int(events[-1] + 60)
    t = np.arange(n, dtype=float)
    task = task_indicator(t, events)

    # Brain: subject-specific response speed and session-specific fluctuations.
    # individuality=0 gives everyone the same brain response and no
    # session-specific fluctuations.
    tau_b = 22.0 + individuality * rng.uniform(-10, 13)
    brain_resp = unit(first_order(task, tau_b))
    brain_true = brain_resp + individuality * 0.35 * slow_noise(n, rng)
    brain_hbo = brain_true + 0.08 * rng.standard_normal(n)

    # Scalp: faster, protocol-specific response, own fluctuations and drift.
    scalp_resp = scalp_sign * unit(first_order(task, rng.uniform(5, 12)))
    drift = rng.normal(0, 0.4) * (t - t.mean()) / n
    scalp_hbo = scalp_resp + 0.6 * slow_noise(n, rng, tau=40) + drift + 0.15 * rng.standard_normal(n)

    # Temple-BF, lagging the brain by BF_LAG_S seconds.
    generic = unit(first_order(task, 22.0))  # what a pure maneuver-follower sees
    tilt_part = generic + 0.35 * slow_noise(n, rng)  # its own, unrelated fluctuations
    person_part = brain_true
    bf_core = (1 - mix) * unit(tilt_part) + mix * unit(person_part)
    bf_core = np.concatenate([np.full(BF_LAG_S, bf_core[0]), bf_core[:-BF_LAG_S]])
    temple_bf = bf_core + 0.3 * slow_noise(n, rng, tau=15) + 0.15 * rng.standard_normal(n)

    sid = f"{protocol}_{idx:02d}"
    return Session(
        session_id=sid,
        subject_id=f"subj_{protocol}_{idx:02d}",
        protocol=protocol,
        events=events,
        time=t,
        signals={"temple_bf": temple_bf, "brain_hbo": brain_hbo, "scalp_hbo": scalp_hbo},
    )


def simulate_dataset(
    mix: float, n_sessions: int = 20, seed: int = 0, protocols=tuple(PROTOCOLS), individuality: float = 1.0
) -> list[Session]:
    rng = np.random.default_rng(seed)
    return [simulate_session(p, i, mix, rng, individuality) for p in protocols for i in range(n_sessions)]


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

    print("Imaginary devices, 20 simulated sessions per protocol, lag-adjusted brain-layer r")
    print(f"{'device':<8}{'protocol':<9}{'real':>7}{'mixed':>7}{'gap':>7}{'p(gap)':>8}"
          f"{'ident':>7}{'res real':>9}{'res mix':>8}")
    for name, mix, ind in (("tilt", 0.0, 1.0), ("person", 1.0, 1.0), ("person*", 1.0, 0.0)):
        res = run(simulate_dataset(mix, seed=seed, individuality=ind), "temple_bf", layers, **kw)
        for protocol, r in res.items():
            s = r["layers"][("full", "brain_hbo", "lag")]
            q = r["layers"][("residual", "brain_hbo", "lag")]
            print(f"{name:<8}{protocol:<9}{s['median_real']:>7.3f}{s['median_mixed']:>7.3f}"
                  f"{s['median_gap']:>7.3f}{fmt_p(s['p_gap_wilcoxon']):>8}"
                  f"{s['identified']:>4}/{s['n']}{q['median_real']:>9.3f}{q['median_mixed']:>8.3f}")

    print("person* = the 'person' device in a world where every brain responds identically")

    print("\nBlending the two devices (w = share of the 'person' signal), head-down tilt")
    print(f"{'w':>5}{'real':>7}{'mixed':>7}{'gap':>7}{'p(gap)':>8}{'ident':>7}")
    for mix in (0.0, 0.25, 0.5, 0.75, 1.0):
        res = run(simulate_dataset(mix, seed=seed, protocols=("hdt",)), "temple_bf", layers,
                  **{**kw, "residual": False})
        s = res["hdt"]["layers"][("full", "brain_hbo", "lag")]
        print(f"{mix:>5.2f}{s['median_real']:>7.3f}{s['median_mixed']:>7.3f}{s['median_gap']:>7.3f}"
              f"{fmt_p(s['p_gap_wilcoxon']):>8}{s['identified']:>4}/{s['n']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Synthetic data for shared_maneuver_test.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="print the tilt-vs-person comparison")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--perms", type=int, default=2000)
    w = sub.add_parser("write", help="write a synthetic dataset to disk")
    w.add_argument("out", type=Path)
    w.add_argument("--device", choices=["tilt", "person"], default="person")
    w.add_argument("--mix", type=float, help="override: share of the person signal, 0..1")
    w.add_argument("--sessions", type=int, default=20, help="sessions per protocol")
    w.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.cmd == "demo":
        demo(args.seed, args.perms)
    else:
        mix = args.mix if args.mix is not None else (1.0 if args.device == "person" else 0.0)
        write_dataset(simulate_dataset(mix, args.sessions, args.seed), args.out)
        print(f"wrote {args.out}/manifest.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
