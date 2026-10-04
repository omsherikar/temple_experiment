#!/usr/bin/env python3
"""Mixed-pair test for the Temple-BF validation study (Gulati et al. 2026).

Compares each session's Temple-BF/NIRS correlation with the correlation
between Temple-BF and other sessions' NIRS from the same protocol. The
difference is the part of the correlation the shared block design does not
explain. Correlations follow the paper's method (Sec. 2.5-2.6).

See README.md for the input format.
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
FS = 1.0
RESPONSE_TAUS_S = (2, 5, 10, 20, 40, 80)


@dataclass
class Session:
    session_id: str
    subject_id: str
    protocol: str
    events: np.ndarray  # block start/end times on the session's own clock
    time: np.ndarray
    signals: dict[str, np.ndarray] = field(default_factory=dict)


def load_manifest(path: Path, bf_col: str, layers: list[str]) -> list[Session]:
    sessions = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            events = np.array([float(x) for x in row["events"].split(";") if x.strip()])
            if len(events) < 2 or np.any(np.diff(events) <= 0):
                raise ValueError(f"{row['session_id']}: events must be >= 2 increasing times")
            time, signals = load_session(path.parent / row["file"], [bf_col, *layers])
            sessions.append(Session(
                session_id=row["session_id"],
                subject_id=row.get("subject_id") or row["session_id"],
                protocol=row["protocol"],
                events=events,
                time=time,
                signals=signals,
            ))
    return sessions


def load_session(path: Path, columns: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
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


def ema_filtfilt(x: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray:
    """Forward-backward exponential moving average."""

    def ema(v):
        return signal.lfilter([alpha], [1.0, alpha - 1.0], v, zi=[(1.0 - alpha) * v[0]])[0]

    return ema(ema(np.asarray(x, float))[::-1])[::-1]


def lowpass1(x: np.ndarray, tau: float) -> np.ndarray:
    """First-order lag with time constant tau (s), starting at x[0]."""
    a = 1.0 - math.exp(-1.0 / (tau * FS))
    return signal.lfilter([a], [1.0, a - 1.0], x, zi=[(1.0 - a) * x[0]])[0]


def interp_finite(t_new: np.ndarray, t: np.ndarray, x: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x)
    return np.interp(t_new, t[ok], x[ok])


def canonical_events(sessions: list[Session]) -> np.ndarray:
    if len({len(s.events) for s in sessions}) != 1:
        raise ValueError("all sessions of a protocol need the same number of events")
    return np.median([s.events - s.events[0] for s in sessions], axis=0)


def block_regressors(t: np.ndarray, events: np.ndarray, task_blocks: tuple[int, ...]) -> np.ndarray:
    """Regressors for any smooth response locked to a session's own blocks.

    A task indicator (1 in task blocks, ramping through the gaps) passed
    through one and two cascaded first-order lags, plus a constant and trend.
    """
    knots_t, knots_v = [], []
    for k in range(len(events) // 2):
        level = 1.0 if (k + 1) in task_blocks else 0.0
        knots_t += [events[2 * k], events[2 * k + 1]]
        knots_v += [level, level]
    if len(events) % 2:
        knots_t.append(events[-1])
        knots_v.append(knots_v[-1] if knots_v else 0.0)
    task = np.interp(t, knots_t, knots_v)
    cols = [np.ones_like(t), (t - t.mean()) / (np.ptp(t) or 1.0), task]
    for tau in RESPONSE_TAUS_S:
        y = task
        for _ in range(2):
            y = lowpass1(y, tau)
            cols.append(y)
    return np.column_stack(cols)


def align_sessions(
    sessions: list[Session],
    align: str,
    smooth: bool,
    residualize: bool = False,
    task_blocks: tuple[int, ...] = (2, 4),
) -> dict[str, np.ndarray]:
    """Resample every session onto a shared 1 Hz protocol clock.

    "warp" stretches each segment between events onto the median timing so
    block onsets line up; "onset" only aligns the first event. Returns
    {signal: array of shape (sessions, samples)}.
    """
    if align == "warp":
        canon = canonical_events(sessions)
        grid = np.arange(0.0, canon[-1] + 1e-9, 1.0 / FS)
    elif align == "onset":
        grid = np.arange(0.0, min(s.events[-1] - s.events[0] for s in sessions) + 1e-9, 1.0 / FS)
    else:
        raise ValueError(align)

    out: dict[str, list[np.ndarray]] = {}
    for s in sessions:
        native = np.arange(s.events[0], s.events[-1] + 1e-9, 1.0 / FS)
        if residualize:
            # Regress out on the session's own clock, before warping, so that
            # its exact transition timing is removed too.
            X = block_regressors(native, s.events, task_blocks)
            if smooth:
                X = np.column_stack([ema_filtfilt(c) for c in X.T])
        session_t = np.interp(grid, canon, s.events) if align == "warp" else grid + s.events[0]
        for name, x in s.signals.items():
            y = interp_finite(native, s.time, x)
            if smooth:
                y = ema_filtfilt(y)
            if residualize:
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                y = y - X @ coef
            out.setdefault(name, []).append(np.interp(session_t, native, y))
    return {k: np.vstack(v) for k, v in out.items()}


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    den = math.sqrt(float(x @ x) * float(y @ y))
    return float(x @ y) / den if den > 0 else np.nan


def lagged_r(bf: np.ndarray, nirs: np.ndarray, max_lag: int = MAX_LAG_S) -> tuple[float, int, float]:
    """(max r over lags, its lag, zero-lag r). Positive lag: NIRS leads BF."""
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


def _unit_rows(a: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def cross_matrix(bf: np.ndarray, nirs: np.ndarray, max_lag: int = MAX_LAG_S) -> tuple[np.ndarray, np.ndarray]:
    """R[i, j] = lagged_r(bf[i], nirs[j]) for all pairs: (max over lags, zero lag)."""
    m = bf.shape[1]
    r_lag = np.full((bf.shape[0], nirs.shape[0]), -np.inf)
    r_zero = None
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            R = _unit_rows(bf[:, lag:]) @ _unit_rows(nirs[:, : m - lag]).T
        else:
            R = _unit_rows(bf[:, : m + lag]) @ _unit_rows(nirs[:, -lag:]).T
        r_lag = np.maximum(r_lag, R)
        if lag == 0:
            r_zero = R
    return r_lag, r_zero


def fisher_mean(r: np.ndarray) -> float:
    r = np.clip(np.asarray(r, float), -0.999999, 0.999999)
    return float(np.tanh(np.mean(np.arctanh(r))))


def mixed_pair_mask(subjects: list[str]) -> np.ndarray:
    subj = np.asarray(subjects)
    return subj[:, None] != subj[None, :]


def random_pairings(mask: np.ndarray, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """Random permutations that only use allowed (mask) pairs."""
    n = mask.shape[0]
    perms = np.empty((n_perm, n), dtype=int)
    k = tries = 0
    while k < n_perm:
        p = rng.permutation(n)
        tries += 1
        if mask[np.arange(n), p].all():
            perms[k] = p
            k += 1
        elif tries > 1000 * n_perm:
            raise RuntimeError("could not sample mixed pairings; too few distinct subjects")
    return perms


def one_sided_wilcoxon(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x != 0)]
    if len(x) == 0:
        return float("nan")
    return float(stats.wilcoxon(x, alternative="greater").pvalue)


def mixed_pair_stats(R: np.ndarray, mask: np.ndarray, perms: np.ndarray) -> dict:
    n = R.shape[0]
    real = np.diag(R).copy()
    # A session's mixed r averages every allowed pair it is part of, as the
    # Temple-BF side (row) and as the NIRS side (column).
    mixed = np.array([fisher_mean(np.concatenate([R[i, mask[i]], R[mask[:, i], i]])) for i in range(n)])
    gap = real - mixed

    z = np.arctanh(np.clip(R, -0.999999, 0.999999))
    t_obs = z[np.arange(n), np.arange(n)].mean()
    t_null = z[np.arange(n)[None, :], perms].mean(axis=1)

    best = np.where(mask | np.eye(n, dtype=bool), R, -np.inf).argmax(axis=1)
    hits = int(np.sum(best == np.arange(n)))
    null_hits = (best[None, :] == perms).sum(axis=1)

    return {
        "n": n,
        "real": real,
        "mixed": mixed,
        "gap": gap,
        "median_real": float(np.median(real)),
        "median_mixed": float(np.median(mixed)),
        "median_gap": float(np.median(gap)),
        "p_gap_wilcoxon": one_sided_wilcoxon(gap),
        "p_pairing_permutation": float((1 + np.sum(t_null >= t_obs)) / (1 + len(t_null))),
        "identified": hits,
        "identified_by_chance": float(np.sum(1.0 / (mask.sum(axis=1) + 1))),
        "p_identification": float((1 + np.sum(null_hits >= hits)) / (1 + len(null_hits))),
    }


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
    mask = mixed_pair_mask([s.subject_id for s in sessions])
    perms = random_pairings(mask, n_perm, rng)

    variants = [("full", align_sessions(sessions, align, smooth))]
    if residual:
        variants.append(("residual", align_sessions(sessions, align, smooth, True, task_blocks)))

    result = {"sessions": [s.session_id for s in sessions], "layers": {}}
    for variant, data in variants:
        for layer in layers:
            r_lag, r_zero = cross_matrix(data[bf_col], data[layer], max_lag)
            for kind, R in (("lag", r_lag), ("zero", r_zero)):
                result["layers"][(variant, layer, kind)] = {"matrix": R, **mixed_pair_stats(R, mask, perms)}

    if len(layers) >= 2:
        a, b = layers[0], layers[1]
        result["contrast"] = {}
        for variant, _ in variants:
            for kind in ("lag", "zero"):
                A = result["layers"][(variant, a, kind)]
                B = result["layers"][(variant, b, kind)]
                contrast = {"layers": (a, b)}
                for part in ("real", "mixed", "gap"):
                    diff = A[part] - B[part]
                    contrast[f"median_{part}_diff"] = float(np.median(diff))
                    contrast[f"p_{part}_diff"] = one_sided_wilcoxon(diff)
                result["contrast"][(variant, kind)] = contrast
    return result


def run(sessions: list[Session], bf_col: str, layers: list[str], **kw) -> dict[str, dict]:
    by_protocol: dict[str, list[Session]] = {}
    for s in sessions:
        by_protocol.setdefault(s.protocol, []).append(s)
    results = {}
    for protocol, group in by_protocol.items():
        if len({s.subject_id for s in group}) < 3:
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
        ("full", "Full sessions"),
        ("residual", "Residual (each session's block-locked response removed)"),
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
        "\nreal: same-session r; mixed: r with other subjects' sessions; gap: real - mixed (medians)"
        "\np(gap): one-sided Wilcoxon on the per-session gap; use the residual one for inference"
        "\np(perm): true vs random pairings (liberal); ident: sessions matched to their own NIRS"
        " (chance in brackets)"
    )
    return "\n".join(lines)


def write_outputs(results: dict[str, dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    with open(out / "per_session.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["protocol", "variant", "layer", "kind", "session_id", "real_r", "mixed_r", "gap"])
        for protocol, res in results.items():
            for (variant, layer, kind), s in res["layers"].items():
                for sid, a, b, g in zip(res["sessions"], s["real"], s["mixed"], s["gap"]):
                    w.writerow([protocol, variant, layer, kind, sid, f"{a:.4f}", f"{b:.4f}", f"{g:.4f}"])
                summary.append({
                    "protocol": protocol, "variant": variant, "layer": layer, "kind": kind,
                    **{k: v for k, v in s.items() if not isinstance(v, np.ndarray)},
                })
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
    ap = argparse.ArgumentParser(description="Mixed-pair test for Temple-BF vs NIRS correlations.")
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--bf", default="temple_bf", help="Temple-BF column")
    ap.add_argument("--layers", default="brain_hbo,scalp_hbo",
                    help="NIRS columns; the first two are contrasted")
    ap.add_argument("--align", choices=["warp", "onset"], default="warp")
    ap.add_argument("--no-smooth", action="store_true")
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--task-blocks", default="2,4", help="1-based task blocks")
    ap.add_argument("--max-lag", type=int, default=MAX_LAG_S)
    ap.add_argument("--perms", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, help="directory for CSV/JSON outputs")
    args = ap.parse_args(argv)

    layers = [c.strip() for c in args.layers.split(",") if c.strip()]
    results = run(
        load_manifest(args.manifest, args.bf, layers),
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
        write_outputs(results, args.out)
        print(f"\nwrote {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
