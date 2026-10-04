#!/usr/bin/env python3
"""Shared-maneuver (mixed-pair) test for the Temple-BF validation study.

Gulati et al. (bioRxiv 2026, doi:10.64898/2026.09.17.751664) correlate
Temple-BF with depth-resolved TD-NIRS dHbO inside each session. Their
Limitation 2 notes that the common block design "can produce high
correlations when two sensors respond to the same imposed maneuver even if
their tissue sources differ". This script measures that component.

For each protocol it correlates every session's Temple-BF trace with every
session's NIRS trace, using the paper's own correlation method, and compares

    real pairs   r(BF_i, NIRS_i)            same session
    mixed pairs  r(BF_i, NIRS_j), j != i    different sessions, same protocol

Mixed pairs share the protocol (same blocks, same order, same durations) but
not the person or the session, so their correlation is what the maneuver
alone produces. The real-minus-mixed gap is the part of the headline
correlation that is specific to the session. This is the "random pair" /
"pseudo-pair" control used in fNIRS hyperscanning and inter-subject
correlation work.

Correlation method, following the paper's Sec. 2.5-2.6:
  * signals interpolated to a common 1 Hz grid over the event-bounded interval
  * forward-backward exponential moving average, alpha = 0.35
  * Pearson r at every integer lag in +/-30 s, the largest r is the
    lag-adjusted correlation (positive lag = NIRS leads Temple-BF);
    the zero-lag r is reported alongside it
  * one-sided Wilcoxon signed-rank tests

Input
-----
A manifest CSV with one row per session:

    session_id,subject_id,protocol,file,events
    s01,p01,hdt,data/s01.csv,0;180;195;375;390;570;585;765;780;960

``events`` lists the block boundaries in seconds on the session's own clock
(start1;end1;start2;end2;...). The first and last define the analysed
interval; all of them are used to align sessions to each other.
``file`` is a CSV (relative to the manifest) with a ``time_s`` column, a
Temple-BF column and one column per NIRS layer, e.g.

    time_s,temple_bf,brain_hbo,scalp_hbo

Run ``python shared_maneuver_test.py --help`` for options.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import signal, stats

EMA_ALPHA = 0.35
MAX_LAG_S = 30
FS = 1.0  # Hz, the paper's common grid


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


@dataclass
class Session:
    session_id: str
    subject_id: str
    protocol: str
    events: np.ndarray  # block boundaries, seconds, session clock
    time: np.ndarray
    signals: dict[str, np.ndarray] = field(default_factory=dict)


def load_manifest(path: Path, bf_col: str, layers: list[str]) -> list[Session]:
    sessions = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            events = np.array([float(x) for x in row["events"].split(";") if x.strip()])
            if len(events) < 2 or np.any(np.diff(events) <= 0):
                raise ValueError(f"{row['session_id']}: events must be >= 2 increasing times")
            time, signals = load_session_file(path.parent / row["file"], [bf_col, *layers])
            sessions.append(
                Session(
                    session_id=row["session_id"],
                    subject_id=row.get("subject_id") or row["session_id"],
                    protocol=row["protocol"],
                    events=events,
                    time=time,
                    signals=signals,
                )
            )
    return sessions


def load_session_file(path: Path, columns: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in ["time_s", *columns] if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        rows = list(reader)
    time = np.array([float(r["time_s"]) for r in rows])
    signals = {c: np.array([float(r[c]) if r[c] != "" else np.nan for r in rows]) for c in columns}
    order = np.argsort(time)
    return time[order], {c: v[order] for c, v in signals.items()}


# --------------------------------------------------------------------------
# Preprocessing (paper Sec. 2.5-2.6)
# --------------------------------------------------------------------------


def ema_filtfilt(x: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray:
    """Forward-backward exponential moving average (zero phase).

    One pass is y[i] = alpha * x[i] + (1 - alpha) * y[i-1], with y[0] = x[0].
    """

    def ema(v):
        return signal.lfilter([alpha], [1.0, alpha - 1.0], v, zi=[(1.0 - alpha) * v[0]])[0]

    return ema(ema(np.asarray(x, float))[::-1])[::-1]


def lowpass1(x: np.ndarray, tau: float) -> np.ndarray:
    """Causal first-order lag with time constant tau (seconds), starting at x[0]."""
    a = 1.0 - math.exp(-1.0 / (tau * FS))
    return signal.lfilter([a], [1.0, a - 1.0], x, zi=[(1.0 - a) * x[0]])[0]


def interp_finite(t_new: np.ndarray, t: np.ndarray, x: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x)
    return np.interp(t_new, t[ok], x[ok])


def canonical_events(sessions: list[Session]) -> np.ndarray:
    """Median event offsets (relative to the first event) across sessions."""
    counts = {len(s.events) for s in sessions}
    if len(counts) != 1:
        raise ValueError("all sessions of a protocol need the same number of events")
    rel = np.array([s.events - s.events[0] for s in sessions])
    return np.median(rel, axis=0)


def to_protocol_time(
    sessions: list[Session],
    align: str,
    smooth: bool,
    residualize: bool = False,
    task_blocks: tuple[int, ...] = (2, 4),
) -> dict[str, np.ndarray]:
    """Put every session of one protocol on a shared 1 Hz protocol clock.

    align="warp": each segment between consecutive events is linearly
        stretched onto the canonical (median) segment, so block onsets
        coincide across sessions. This is the conservative choice: it gives
        mixed pairs the best possible chance of matching.
    align="onset": sessions are only shifted so the first event is t=0 and
        cropped to the shortest session.

    residualize=True removes, before alignment, everything a session's own
    block timing can explain (see ``maneuver_design``).

    Returns {signal name: array (n_sessions, n_samples)}.
    """
    if align == "warp":
        canon = canonical_events(sessions)
        grid = np.arange(0.0, canon[-1] + 1e-9, 1.0 / FS)
    elif align == "onset":
        dur = min(s.events[-1] - s.events[0] for s in sessions)
        grid = np.arange(0.0, dur + 1e-9, 1.0 / FS)
    else:
        raise ValueError(align)

    out: dict[str, list[np.ndarray]] = {}
    for s in sessions:
        # Paper: 1 Hz grid over the shared event-bounded interval, then smooth.
        native = np.arange(s.events[0], s.events[-1] + 1e-9, 1.0 / FS)
        if residualize:
            X = maneuver_design(native, s.events, task_blocks)
            if smooth:  # same linear filter on regressors as on the signals
                X = np.column_stack([ema_filtfilt(c) for c in X.T])
        for name, x in s.signals.items():
            y = interp_finite(native, s.time, x)
            if smooth:
                y = ema_filtfilt(y)
            if residualize:
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                y = y - X @ coef
            if align == "warp":
                session_t = np.interp(grid, canon, s.events)
            else:
                session_t = grid + s.events[0]
            out.setdefault(name, []).append(np.interp(session_t, native, y))
    return {k: np.vstack(v) for k, v in out.items()}


# --------------------------------------------------------------------------
# Correlation (paper Sec. 2.6)
# --------------------------------------------------------------------------


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    den = math.sqrt(float(x @ x) * float(y @ y))
    return float(x @ y) / den if den > 0 else np.nan


def lagged_r(bf: np.ndarray, nirs: np.ndarray, max_lag: int = MAX_LAG_S) -> tuple[float, int, float]:
    """Return (lag-adjusted r, optimal lag in samples, zero-lag r).

    Positive lag means the NIRS trace leads Temple-BF: BF(t) is compared with
    NIRS(t - lag). The lag-adjusted r is the largest r over all lags.
    """
    n = len(bf)
    best_r, best_lag = -np.inf, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            r = _pearson(nirs[: n - lag], bf[lag:])
        else:
            r = _pearson(nirs[-lag:], bf[: n + lag])
        if r > best_r:
            best_r, best_lag = r, lag
    return best_r, best_lag, _pearson(nirs, bf)


def cross_matrix(bf: np.ndarray, nirs: np.ndarray, max_lag: int = MAX_LAG_S) -> tuple[np.ndarray, np.ndarray]:
    """R[i, j] = r(BF of session i, NIRS of session j); lag-adjusted and zero-lag.

    Same definition as ``lagged_r``, computed for all pairs at once.
    """
    m = bf.shape[1]

    def z(a):
        a = a - a.mean(axis=1, keepdims=True)
        return a / np.linalg.norm(a, axis=1, keepdims=True)

    r_lag = np.full((bf.shape[0], nirs.shape[0]), -np.inf)
    r_zero = None
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            R = z(bf[:, lag:]) @ z(nirs[:, : m - lag]).T
        else:
            R = z(bf[:, : m + lag]) @ z(nirs[:, -lag:]).T
        r_lag = np.maximum(r_lag, R)
        if lag == 0:
            r_zero = R
    return r_lag, r_zero


def fisher_mean(r: np.ndarray) -> float:
    r = np.clip(np.asarray(r, float), -0.999999, 0.999999)
    return float(np.tanh(np.mean(np.arctanh(r))))


# --------------------------------------------------------------------------
# Mixed-pair statistics
# --------------------------------------------------------------------------


def eligible_mask(subjects: list[str]) -> np.ndarray:
    """True where (i, j) is a valid mixed pair: different session AND subject."""
    subj = np.asarray(subjects)
    return subj[:, None] != subj[None, :]


def random_derangements(mask: np.ndarray, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """Random permutations pi with mask[i, pi(i)] True for all i (rejection sampling)."""
    n = mask.shape[0]
    perms = np.empty((n_perm, n), dtype=int)
    k = 0
    tries = 0
    while k < n_perm:
        p = rng.permutation(n)
        tries += 1
        if mask[np.arange(n), p].all():
            perms[k] = p
            k += 1
        elif tries > 1000 * n_perm:
            raise RuntimeError("could not sample valid mixed pairings; too few distinct subjects")
    return perms


def one_sided_wilcoxon(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x != 0)]
    if len(x) == 0:
        return float("nan")
    return float(stats.wilcoxon(x, alternative="greater").pvalue)


def mixed_pair_stats(R: np.ndarray, mask: np.ndarray, perms: np.ndarray) -> dict:
    """Compare real pairs (diagonal) with mixed pairs (masked off-diagonal).

    Per session i, the mixed-pair r is the Fisher-z mean over every valid
    mixed pair that involves session i in either role: BF_i with other
    sessions' NIRS (row) and NIRS_i with other sessions' BF (column).
    """
    n = R.shape[0]
    real = np.diag(R).copy()
    mixed = np.array(
        [fisher_mean(np.concatenate([R[i, mask[i]], R[mask[:, i], i]])) for i in range(n)]
    )
    gap = real - mixed

    # Permutation test on the pairing itself: is the true pairing better than
    # random valid pairings? Statistic = Fisher-mean r over the n pairs.
    z = np.arctanh(np.clip(R, -0.999999, 0.999999))
    t_obs = z[np.arange(n), np.arange(n)].mean()
    t_null = z[np.arange(n)[None, :], perms].mean(axis=1)
    p_perm = (1 + np.sum(t_null >= t_obs)) / (1 + len(t_null))

    # Identification: does BF_i correlate best with its own session's NIRS?
    candidates = np.where(mask | np.eye(n, dtype=bool), R, -np.inf)
    best = candidates.argmax(axis=1)
    hits = int(np.sum(best == np.arange(n)))
    chance = float(np.sum(1.0 / (mask.sum(axis=1) + 1)))
    null_hits = (best[None, :] == perms).sum(axis=1)
    p_ident = (1 + np.sum(null_hits >= hits)) / (1 + len(null_hits))

    return {
        "n": n,
        "real": real,
        "mixed": mixed,
        "gap": gap,
        "median_real": float(np.median(real)),
        "median_mixed": float(np.median(mixed)),
        "median_gap": float(np.median(gap)),
        "p_gap_wilcoxon": one_sided_wilcoxon(gap),
        "p_pairing_permutation": float(p_perm),
        "identified": hits,
        "identified_by_chance": chance,
        "p_identification": float(p_ident),
        "mixed_share_of_real": float(np.median(mixed) / np.median(real)) if np.median(real) else np.nan,
    }


MANEUVER_TAUS_S = (2, 5, 10, 20, 40, 80)


def maneuver_design(t: np.ndarray, events: np.ndarray, task_blocks: tuple[int, ...]) -> np.ndarray:
    """Regressors for any smooth response locked to this session's own blocks.

    The task indicator is 1 during the task blocks (1-based block numbers,
    default 2 and 4: tilt/squat/supine), 0 during the others, and ramps
    linearly through the transition gaps. It is passed through first- and
    second-order lags (one and two cascaded exponentials) with time
    constants from 2 to 80 s, which spans fast responses, the slow 60-90 s
    rise the paper reports for brain dHbO, and the ~13 s Temple-BF delay.
    A constant and a linear trend are included.

    Built from the session's own event times, so it also captures that
    session's exact transition timing - the part of the maneuver that real
    pairs share and mixed pairs do not.
    """
    n_blocks = len(events) // 2
    knots_t, knots_v = [], []
    for k in range(n_blocks):
        level = 1.0 if (k + 1) in task_blocks else 0.0
        knots_t += [events[2 * k], events[2 * k + 1]]
        knots_v += [level, level]
    if len(events) % 2:  # odd count: treat the last event as the end of the last block
        knots_t.append(events[-1])
        knots_v.append(knots_v[-1] if knots_v else 0.0)
    task = np.interp(t, knots_t, knots_v)
    cols = [np.ones_like(t), (t - t.mean()) / (np.ptp(t) or 1.0), task]
    for tau in MANEUVER_TAUS_S:
        y = task
        for _ in range(2):
            y = lowpass1(y, tau)
            cols.append(y)
    return np.column_stack(cols)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def analyse_protocol(
    sessions: list[Session],
    bf_col: str,
    layers: list[str],
    align: str = "warp",
    smooth: bool = True,
    n_perm: int = 10000,
    seed: int = 0,
    residual: bool = True,
    max_lag: int = MAX_LAG_S,
    task_blocks: tuple[int, ...] = (2, 4),
) -> dict:
    rng = np.random.default_rng(seed)
    aligned = to_protocol_time(sessions, align, smooth)
    subjects = [s.subject_id for s in sessions]
    mask = eligible_mask(subjects)
    perms = random_derangements(mask, n_perm, rng)

    result = {"sessions": [s.session_id for s in sessions], "layers": {}}
    variants = [("full", aligned)]
    if residual:
        variants.append(("residual", to_protocol_time(sessions, align, smooth, True, task_blocks)))

    for variant, data in variants:
        for layer in layers:
            r_lag, r_zero = cross_matrix(data[bf_col], data[layer], max_lag)
            for kind, R in (("lag", r_lag), ("zero", r_zero)):
                result["layers"][(variant, layer, kind)] = {
                    "matrix": R,
                    **mixed_pair_stats(R, mask, perms),
                }

    # Does the brain layer keep its advantage over the scalp layer once the
    # shared-maneuver component is subtracted?  (Only if both are present.)
    if len(layers) >= 2:
        a, b = layers[0], layers[1]
        result["contrast"] = {}
        for variant, _ in variants:
            for kind in ("lag", "zero"):
                A = result["layers"][(variant, a, kind)]
                B = result["layers"][(variant, b, kind)]
                result["contrast"][(variant, kind)] = {
                    "layers": (a, b),
                    "median_real_diff": float(np.median(A["real"] - B["real"])),
                    "p_real_diff": one_sided_wilcoxon(A["real"] - B["real"]),
                    "median_mixed_diff": float(np.median(A["mixed"] - B["mixed"])),
                    "p_mixed_diff": one_sided_wilcoxon(A["mixed"] - B["mixed"]),
                    "median_gap_diff": float(np.median(A["gap"] - B["gap"])),
                    "p_gap_diff": one_sided_wilcoxon(A["gap"] - B["gap"]),
                }
    return result


def run(sessions: list[Session], bf_col: str, layers: list[str], **kw) -> dict[str, dict]:
    by_protocol: dict[str, list[Session]] = {}
    for s in sessions:
        by_protocol.setdefault(s.protocol, []).append(s)
    results = {}
    for protocol, group in by_protocol.items():
        n_subj = len({s.subject_id for s in group})
        if n_subj < 3:
            print(f"skipping {protocol}: needs sessions from >= 3 subjects", file=sys.stderr)
            continue
        results[protocol] = analyse_protocol(group, bf_col, layers, **kw)
    return results


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def report(results: dict[str, dict], layers: list[str]) -> str:
    lines = []
    for variant, title in (
        ("full", "Full-session correlations (the paper's analysis; describes the headline)"),
        ("residual", "After regressing out each session's own block-locked response "
                     "(p(gap) here is the calibrated test for session-specific tracking)"),
    ):
        if not any((variant, layers[0], "lag") in r["layers"] for r in results.values()):
            continue
        lines.append(f"\n== {title} ==")
        for kind, label in (("lag", "lag-adjusted, +/-30 s"), ("zero", "zero-lag")):
            lines.append(f"\n-- {label} --")
            lines.append(
                f"{'protocol':<10}{'layer':<12}{'n':>3}{'real':>8}{'mixed':>8}{'gap':>8}"
                f"{'p(gap)':>9}{'p(perm)':>9}{'ident':>9}{'p(id)':>8}"
            )
            for protocol, res in results.items():
                for layer in layers:
                    s = res["layers"][(variant, layer, kind)]
                    lines.append(
                        f"{protocol:<10}{layer:<12}{s['n']:>3}{s['median_real']:>8.3f}"
                        f"{s['median_mixed']:>8.3f}{s['median_gap']:>8.3f}"
                        f"{fmt_p(s['p_gap_wilcoxon']):>9}{fmt_p(s['p_pairing_permutation']):>9}"
                        f"{s['identified']:>4}/{s['n']:<2}({s['identified_by_chance']:.0f})"
                        f"{fmt_p(s['p_identification']):>8}"
                    )
            if len(layers) >= 2:
                a, b = layers[0], layers[1]
                lines.append(f"\n{a} minus {b} (one-sided Wilcoxon, {a} > {b}):")
                lines.append(f"{'protocol':<10}{'real':>16}{'mixed':>16}{'gap':>16}")
                for protocol, res in results.items():
                    c = res["contrast"][(variant, kind)]
                    lines.append(
                        f"{protocol:<10}"
                        f"{c['median_real_diff']:>8.3f} p={fmt_p(c['p_real_diff']):<6}"
                        f"{c['median_mixed_diff']:>8.3f} p={fmt_p(c['p_mixed_diff']):<6}"
                        f"{c['median_gap_diff']:>8.3f} p={fmt_p(c['p_gap_diff']):<6}"
                    )
    lines.append(
        "\nreal  = median r of same-session pairs; mixed = median per-session Fisher-mean r of"
        "\n        different-subject pairs in the same protocol; gap = median per-session real - mixed"
        "\np(gap) = one-sided Wilcoxon signed-rank, gap > 0. Use the residual p(gap) for inference:"
        "\n         on full sessions, real pairs also share their own transition timing, which"
        "\n         inflates false positives when transition durations vary (see README, Calibration)"
        "\np(perm)= permutation test: true pairing vs random different-subject pairings (liberal)"
        "\nident = sessions whose BF matches its own NIRS best (expected by chance in brackets)"
    )
    return "\n".join(lines)


def write_outputs(results: dict[str, dict], layers: list[str], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_session.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["protocol", "variant", "layer", "kind", "session_id", "real_r", "mixed_r", "gap"])
        for protocol, res in results.items():
            for (variant, layer, kind), s in res["layers"].items():
                for sid, a, b, g in zip(res["sessions"], s["real"], s["mixed"], s["gap"]):
                    w.writerow([protocol, variant, layer, kind, sid, f"{a:.4f}", f"{b:.4f}", f"{g:.4f}"])
    summary = []
    for protocol, res in results.items():
        for (variant, layer, kind), s in res["layers"].items():
            summary.append(
                {
                    "protocol": protocol,
                    "variant": variant,
                    "layer": layer,
                    "kind": kind,
                    **{k: v for k, v in s.items() if not isinstance(v, np.ndarray)},
                }
            )
            np.savetxt(
                out / f"matrix_{protocol}_{variant}_{layer}_{kind}.csv",
                s["matrix"],
                delimiter=",",
                fmt="%.4f",
                header="rows: Temple-BF of session i; cols: NIRS of session j; order: "
                + ";".join(res["sessions"]),
            )
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("manifest", type=Path, help="manifest CSV (see module docstring)")
    ap.add_argument("--bf", default="temple_bf", help="Temple-BF column name")
    ap.add_argument(
        "--layers",
        default="brain_hbo,scalp_hbo",
        help="comma-separated NIRS columns; the first two are contrasted (default: brain_hbo,scalp_hbo)",
    )
    ap.add_argument("--align", choices=["warp", "onset"], default="warp",
                    help="how sessions are put on a shared protocol clock (default: warp)")
    ap.add_argument("--no-smooth", action="store_true", help="skip the alpha=0.35 forward-backward EMA")
    ap.add_argument("--no-residual", action="store_true", help="skip the residual (template-removed) analysis")
    ap.add_argument("--task-blocks", default="2,4",
                    help="1-based task blocks for the residual analysis (default 2,4)")
    ap.add_argument("--max-lag", type=int, default=MAX_LAG_S, help="lag search window in seconds (default 30)")
    ap.add_argument("--perms", type=int, default=10000, help="random pairings for permutation tests")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, help="directory for per-session CSV, summary JSON and r matrices")
    args = ap.parse_args(argv)

    layers = [c.strip() for c in args.layers.split(",") if c.strip()]
    sessions = load_manifest(args.manifest, args.bf, layers)
    results = run(
        sessions,
        args.bf,
        layers,
        align=args.align,
        smooth=not args.no_smooth,
        n_perm=args.perms,
        seed=args.seed,
        residual=not args.no_residual,
        max_lag=args.max_lag,
        task_blocks=tuple(int(b) for b in args.task_blocks.split(",") if b.strip()),
    )
    print(report(results, layers))
    if args.out:
        write_outputs(results, layers, args.out)
        print(f"\nwrote {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
