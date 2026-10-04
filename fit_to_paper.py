#!/usr/bin/env python3
"""Fit the simulator to the statistics published by Gulati et al. 2026.

The simulator has a few parameters that no paper reports: how large the
session-specific brain variation is, how noisy Temple-BF is, and the scalp
response. Instead of choosing them by hand, this script fits them so that the
simulated sessions, analysed with the paper's method, reproduce the paper's
published numbers:

    Table 1  median lag-adjusted r, brain and scalp layer
    Table 1  median zero-lag r, brain layer
    Table 2  median transition-window r (Fisher-averaged within session)
    Fig. 4   r between the group-averaged traces, brain and scalp layer

Agreement is judged against the paper's own sampling error: the standard
error of each statistic is estimated from 30 simulated 20-session studies,
and a parameter set is "consistent with the paper" when the chi-square
distance sum(((z_sim - z_paper) / SE)^2) over the four brain-layer
statistics and the median lag is below the 95% point of chi-square with 5
degrees of freedom (11.07).

The key unknown, how much of Temple-BF follows the session's own brain
physiology rather than the maneuver (``mix``, w), is NOT fitted. For each w
on a grid the script:

  1. finds the best-fitting parameters and whether any are consistent with
     the paper (can the published statistics rule this w out?);
  2. searches the consistent parameter sets for the smallest and the largest
     real-minus-mixed gap (an identification set: every gap in between is
     compatible with what the paper reports);
  3. runs the full mixed-pair test on 40 simulated studies at both ends, to
     give the power of the test with 20 sessions.

    python fit_to_paper.py                       # writes paper_fit.json
    python fit_to_paper.py --mixes 1 --phys-share 1 --out results/sens_q100.json
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy import optimize

from shared_maneuver_test import (
    canonical_events,
    cross_matrix,
    fisher_mean,
    lagged_r,
    run,
    to_protocol_time,
)
from simulate import FIT_FILE, SimParams, default_params, simulate_dataset

# Gulati et al. 2026, n = 20 sessions per protocol.
PAPER_TARGETS = {
    "hdt": {
        "r_lag_brain": 0.912,  # Table 1
        "r_lag_scalp": 0.494,  # Table 1
        "r_zero_brain": 0.826,  # Table 1
        "r_transition": 0.878,  # Table 2
        "r_group_brain": 0.915,  # Fig. 4A
        "r_group_scalp": 0.430,  # Fig. 4A
    },
    "squat": {
        "r_lag_brain": 0.843,
        "r_lag_scalp": 0.700,
        "r_zero_brain": 0.701,
        "r_transition": 0.861,
        "r_group_brain": 0.916,  # Fig. 4B
        "r_group_scalp": 0.783,
    },
    "supine": {
        "r_lag_brain": 0.839,
        "r_lag_scalp": 0.767,
        "r_zero_brain": 0.796,
        "r_transition": 0.876,
        "r_group_brain": 0.941,  # Fig. 4C
        "r_group_scalp": 0.929,
    },
}
# Sec. 2.6: transition windows of 90 s (head-down tilt, stand-to-supine) and
# 45 s (stand-to-squat) either side of each baseline-to-task transition.
TRANSITION_HALF_WINDOW_S = {"hdt": 90, "squat": 45, "supine": 90}
# Sec. 3.2: median optimal brain lag 13-15 s (reported as a check, not fitted).
PAPER_LAG_RANGE_S = (13, 15)

FIT_SEEDS = (1000, 1001, 1002, 1003)  # common random numbers: 4 x 20 sessions
SE_SEEDS = tuple(range(2000, 2030))  # 30 independent studies for standard errors
CHI2_95_DF5 = 11.07  # 95% point of chi-square, 5 df (4 brain statistics + lag)
BRAIN_KEYS = ("r_lag_brain", "r_zero_brain", "r_transition", "r_group_brain")
SCALP_KEYS = ("r_lag_scalp", "r_group_scalp")


def paired_lagged_r(bf: np.ndarray, nirs: np.ndarray, max_lag: int = 30):
    """Row-wise lag-adjusted r, optimal lag and zero-lag r for same-session pairs."""
    m = bf.shape[1]

    def z(a):
        a = a - a.mean(axis=1, keepdims=True)
        return a / np.linalg.norm(a, axis=1, keepdims=True)

    best = np.full(bf.shape[0], -np.inf)
    best_lag = np.zeros(bf.shape[0], int)
    r0 = None
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            r = np.sum(z(bf[:, lag:]) * z(nirs[:, : m - lag]), axis=1)
        else:
            r = np.sum(z(bf[:, : m + lag]) * z(nirs[:, -lag:]), axis=1)
        better = r > best
        best[better], best_lag[better] = r[better], lag
        if lag == 0:
            r0 = r
    return best, best_lag, r0


def paper_statistics(sessions, protocol: str) -> dict[str, float]:
    """The paper's published statistics, computed on a set of sessions."""
    data = to_protocol_time(sessions, "warp", smooth=True)
    canon = canonical_events(sessions)
    bf, brain, scalp = data["temple_bf"], data["brain_hbo"], data["scalp_hbo"]

    r_lag_b, lag_b, r0_b = paired_lagged_r(bf, brain)
    r_lag_s, _, _ = paired_lagged_r(bf, scalp)

    # Transition windows around the two baseline-to-task transitions (end of
    # blocks 1 and 3), Fisher-averaged within session. Temple-BF is aligned
    # with the session's optimal lag first. The paper does not say so for this
    # analysis (it does for its partial correlations), but its numbers require
    # it: for stand-to-squat the reported transition r (0.861) is above the
    # zero-lag full-session r (0.701), while a 14 s lag alone caps a zero-lag
    # +/-45 s window at about 0.88 even with no noise at all.
    half = TRANSITION_HALF_WINDOW_S[protocol]
    m = bf.shape[1]
    zs = []
    for onset in (canon[1], canon[5]):
        r = []
        for b, nv, lag in zip(bf, brain, lag_b):
            lo = max(int(round(onset - half)), -lag, 0)
            hi = min(int(round(onset + half)), m - lag, m)
            r.append(np.corrcoef(b[lo + lag: hi + lag], nv[lo:hi])[0, 1])
        zs.append(np.arctanh(np.clip(r, -0.999999, 0.999999)))
    r_trans = np.tanh(np.mean(zs, axis=0))

    # Mixed-pair gap (lag-adjusted, brain layer), as in shared_maneuver_test:
    # real r minus the Fisher-mean r of every other session in either role.
    R, _ = cross_matrix(bf, brain)
    n = R.shape[0]
    off = ~np.eye(n, dtype=bool)
    mixed = np.array([fisher_mean(np.concatenate([R[i, off[i]], R[off[:, i], i]])) for i in range(n)])
    gap = np.diag(R) - mixed

    return {
        "r_lag_brain": float(np.median(r_lag_b)),
        "r_lag_scalp": float(np.median(r_lag_s)),
        "r_zero_brain": float(np.median(r0_b)),
        "r_transition": float(np.median(r_trans)),
        "r_group_brain": lagged_r(bf.mean(0), brain.mean(0))[0],
        "r_group_scalp": lagged_r(bf.mean(0), scalp.mean(0))[0],
        "lag_brain_s": float(np.median(lag_b)),
        "gap": float(np.median(gap)),
        "identified": float(np.sum(R.argmax(axis=1) == np.arange(n))),
    }


def statistics_over_seeds(params: SimParams, protocol: str, mix: float, seeds=FIT_SEEDS) -> dict[str, float]:
    per_seed = [
        paper_statistics(simulate_dataset(mix, {protocol: params}, seed=s, protocols=(protocol,)), protocol)
        for s in seeds
    ]
    return {k: float(np.mean([d[k] for d in per_seed])) for k in per_seed[0]}


def _z(r: float) -> float:
    return math.atanh(max(min(r, 0.999999), -0.999999))


def raw_loss(stats: dict[str, float], targets: dict[str, float], keys) -> float:
    """Squared error in Fisher z, plus the median lag in units of 5 s."""
    err = sum((_z(stats[k]) - _z(targets[k])) ** 2 for k in keys)
    if "r_lag_brain" in keys:
        err += ((stats["lag_brain_s"] - np.mean(PAPER_LAG_RANGE_S)) / 5.0) ** 2
    return err


def standard_errors(params: SimParams, protocol: str, mix: float) -> dict[str, float]:
    """Sampling SD of each statistic across independent 20-session studies
    (Fisher z for correlations, seconds for the lag)."""
    per = [paper_statistics(simulate_dataset(mix, {protocol: params}, seed=s, protocols=(protocol,)), protocol)
           for s in SE_SEEDS]
    se = {k: float(np.std([_z(d[k]) for d in per], ddof=1)) for k in BRAIN_KEYS + SCALP_KEYS}
    # The paper reports the median lag only as "13 to 15 s"; do not ask for
    # more precision than that.
    se["lag_brain_s"] = max(float(np.std([d["lag_brain_s"] for d in per], ddof=1)), 1.0)
    return se


def chi2(stats: dict[str, float], targets: dict[str, float], se: dict[str, float]) -> float:
    err = sum(((_z(stats[k]) - _z(targets[k])) / se[k]) ** 2 for k in BRAIN_KEYS)
    return err + ((stats["lag_brain_s"] - np.mean(PAPER_LAG_RANGE_S)) / se["lag_brain_s"]) ** 2


SEARCH_EFFORT = 1.0  # multiplies every differential-evolution budget (--search-effort)


def _set_effort(effort: float) -> None:
    global SEARCH_EFFORT
    SEARCH_EFFORT = effort


def _de(objective, bounds, x0=None, seed=0, maxiter=30):
    """Differential evolution (global, gradient-free). The objective is
    deterministic (common random numbers) but not smooth - it is built from
    medians and lag maxima - so local simplex searches get stuck."""
    return optimize.differential_evolution(
        objective, bounds, x0=x0, seed=seed, popsize=8, maxiter=int(maxiter * SEARCH_EFFORT), tol=1e-4,
        polish=False,
    )


# Stage-1 parameters: (name, lower, upper, searched on a log scale).
BRAIN_SPEC = [
    ("brain_noise", 0.01, 3.0, True),
    ("bf_noise", 0.01, 3.0, True),
    ("bf_lag_s", 0.0, 30.0, False),
    ("bf_drift", -3.0, 3.0, False),
]
# Extra mechanisms fitted only where the base model cannot reproduce the
# paper. For head-down tilt no version of the base model comes close
# (chi-square 400-580); a screen of candidate mechanisms
# (results/hdt_screen.txt) found that only models with a Temple-BF
# transient at each transition get close. Temple-BF includes heart-rate
# features (Sec. 2.2.1) and heart rate changes abruptly at every tilt
# transition (Fig. 4A). The brain-response shape is bounded by Fig. 3A: in
# head-down tilt brain dHbO rises almost at once (fast share free) and
# declines only ~12% within a block (adaptation capped at 25%).
EXTRA_SPEC: dict[str, list] = {
    "hdt": [
        ("brain_fast_share", 0.0, 1.0, False),
        ("brain_adapt", 0.0, 0.25, False),
        ("brain_adapt_tau_s", 10.0, 200.0, False),
        ("bf_transient", -2.0, 2.0, False),
    ],
}
SCALP_BOUNDS = [(-5.0, 5.0), (math.log(1.0), math.log(200.0))]


def _spec(protocol: str) -> list:
    return BRAIN_SPEC + EXTRA_SPEC.get(protocol, [])


def _bounds(spec) -> list:
    return [(math.log(lo), math.log(hi)) if lg else (lo, hi) for _, lo, hi, lg in spec]


def _brain(base: SimParams, x, spec) -> SimParams:
    return replace(base, **{n: float(np.exp(v)) if lg else float(v) for (n, _, _, lg), v in zip(spec, x)})


def _brain_x(p: SimParams, spec) -> np.ndarray:
    return np.array([math.log(getattr(p, n)) if lg else getattr(p, n) for n, _, _, lg in spec])


def fit_one(job: tuple[str, float, SimParams, bool]) -> dict:
    protocol, mix, base, with_bounds = job
    targets = PAPER_TARGETS[protocol]
    stats = lambda p: statistics_over_seeds(p, protocol, mix)  # noqa: E731
    spec = _spec(protocol)
    bounds = _bounds(spec)

    # 1. Point fit in Fisher-z units, then standard errors at that point,
    #    then refit in standard-error units.
    p0 = _brain(base, _de(lambda x: raw_loss(stats(_brain(base, x, spec)), targets, BRAIN_KEYS), bounds).x, spec)
    se = standard_errors(p0, protocol, mix)
    obj = lambda x: chi2(stats(_brain(base, x, spec)), targets, se)  # noqa: E731
    best = _de(obj, bounds, x0=_brain_x(p0, spec), maxiter=20)
    p1 = _brain(base, best.x, spec)
    chi2_min = float(best.fun)
    consistent = chi2_min <= CHI2_95_DF5

    # 2. Scalp parameters (they do not affect the brain statistics or the gap).
    def scalp(p, x):
        return replace(p, scalp_amp=float(x[0]), scalp_tau_s=float(np.exp(x[1])))

    s2 = _de(lambda x: raw_loss(stats(scalp(p1, x)), targets, SCALP_KEYS), SCALP_BOUNDS)
    scalp_x = s2.x
    p1 = scalp(p1, scalp_x)

    # 3. Smallest and largest gap among parameter sets consistent with the
    #    paper (or, if none is, within the best fit's chi-square).
    limit = max(CHI2_95_DF5, chi2_min)
    ends = {}
    for name, sign in (("min", 1.0), ("max", -1.0)):
        if not with_bounds:
            ends[name] = p1
            continue

        def bound_obj(x, sign=sign):
            st = stats(_brain(base, x, spec))
            return sign * st["gap"] + 100.0 * max(0.0, chi2(st, targets, se) - limit)

        res = _de(bound_obj, bounds, x0=_brain_x(p1, spec), maxiter=25)
        ends[name] = scalp(_brain(base, res.x, spec), scalp_x)

    out = {
        "protocol": protocol,
        "mix": mix,
        "params": asdict(p1),
        "chi2": chi2_min,
        "consistent": consistent,
        "se": se,
        "fitted_stats": stats(p1),
        "scalp_loss": float(s2.fun),
        "targets": targets,
        "gap_bounds": {},
    }
    for name, p in ends.items():
        st = stats(p)
        out["gap_bounds"][name] = {"params": asdict(p), "stats": st, "chi2": chi2(st, targets, se)}
    return out


def predict_one(job: tuple[str, float, dict, int]) -> dict:
    """Run the mixed-pair test on many simulated 20-session studies."""
    protocol, mix, params, n_studies = job
    p = SimParams(**{k: tuple(v) if isinstance(v, list) else v for k, v in params.items()})
    rows = []
    for k in range(n_studies):
        sessions = simulate_dataset(mix, {protocol: p}, seed=5000 + k, protocols=(protocol,))
        res = run(sessions, "temple_bf", ["brain_hbo", "scalp_hbo"], n_perm=200, seed=k)[protocol]
        full = res["layers"][("full", "brain_hbo", "lag")]
        resid = res["layers"][("residual", "brain_hbo", "lag")]
        contrast = res["contrast"][("full", "lag")]
        rows.append([
            full["median_real"], full["median_mixed"], full["median_gap"], full["p_gap_wilcoxon"] < 0.05,
            full["identified"], resid["median_gap"], resid["p_gap_wilcoxon"] < 0.05,
            contrast["p_real_diff"] < 0.05, contrast["p_gap_diff"] < 0.05,
        ])
    a = np.array(rows, float)
    return {
        "studies": n_studies,
        "real": float(np.median(a[:, 0])),
        "mixed": float(np.median(a[:, 1])),
        "gap": float(np.median(a[:, 2])),
        "power_gap": float(a[:, 3].mean()),
        "identified": float(np.median(a[:, 4])),
        "resid_gap": float(np.median(a[:, 5])),
        "power_resid_gap": float(a[:, 6].mean()),
        "share_brain_gt_scalp_real": float(a[:, 7].mean()),
        "share_brain_gt_scalp_gap": float(a[:, 8].mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mixes", default="0,0.5,1", help="device mixes w (0 = tilt, 1 = person)")
    ap.add_argument("--protocols", default="hdt,squat,supine")
    ap.add_argument("--phys-share", type=float, default=0.5,
                    help="assumed physiological share of session-specific brain variance")
    ap.add_argument("--band-low", type=float, default=0.01, help="lower edge of the fluctuation band, Hz")
    ap.add_argument("--gap", default="8,20", help="assumed transition gap range, s")
    ap.add_argument("--studies", type=int, default=40, help="simulated studies per prediction")
    ap.add_argument("--search-effort", type=float, default=1.0,
                    help="multiply the optimiser budget (use >1 for protocols with extra parameters)")
    ap.add_argument("--out", type=Path, default=FIT_FILE)
    args = ap.parse_args()
    global SEARCH_EFFORT
    SEARCH_EFFORT = args.search_effort

    mixes = [float(m) for m in args.mixes.split(",")]
    protocols = args.protocols.split(",")
    gap = tuple(float(g) for g in args.gap.split(","))
    bases = {
        p: replace(b, phys_share=args.phys_share, band_hz=(args.band_low, b.band_hz[1]), gap_s=gap)
        for p, b in default_params().items()
    }

    # w = 0 has no session-specific tracking, so its gap is zero by
    # construction; only its fit quality is of interest.
    jobs = [(p, m, bases[p], m > 0) for p in protocols for m in mixes]
    with ProcessPoolExecutor(initializer=_set_effort, initargs=(SEARCH_EFFORT,)) as ex:
        fits = list(ex.map(fit_one, jobs))
        pred_jobs = [(f["protocol"], f["mix"], f["gap_bounds"][e]["params"], args.studies)
                     for f in fits for e in ("min", "max")]
        preds = list(ex.map(predict_one, pred_jobs))
    for i, f in enumerate(fits):
        f["gap_bounds"]["min"]["prediction"] = preds[2 * i]
        f["gap_bounds"]["max"]["prediction"] = preds[2 * i + 1]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "settings": {"phys_share": args.phys_share, "band_hz": [args.band_low, 0.12], "gap_s": list(gap),
                     "fit_seeds": list(FIT_SEEDS), "se_seeds": [SE_SEEDS[0], SE_SEEDS[-1]],
                     "chi2_threshold": CHI2_95_DF5, "studies": args.studies},
        "fits": fits,
    }, indent=2))
    print(report(fits, args.studies))
    print(f"\nwrote {args.out}")


def report(fits: list[dict], studies: int) -> str:
    keys = list(PAPER_TARGETS["hdt"])
    short = ["lag b", "lag s", "zero b", "trans", "grp b", "grp s"]
    lines = ["Best fit to the paper's statistics (simulated values; paper in the first row)",
             f"chi2 = distance in standard-error units, 5 df; consistent if <= {CHI2_95_DF5}"]
    for protocol in dict.fromkeys(f["protocol"] for f in fits):
        t = PAPER_TARGETS[protocol]
        lines.append(f"\n{protocol:<8}{'w':>5}" + "".join(f"{s:>8}" for s in short) + f"{'lag s':>7}{'chi2':>7}")
        lines.append(f"{'paper':<8}{'':>5}" + "".join(f"{t[k]:>8.3f}" for k in keys) + f"{'13-15':>7}")
        for f in fits:
            if f["protocol"] == protocol:
                st = f["fitted_stats"]
                lines.append(f"{'':<8}{f['mix']:>5.2f}" + "".join(f"{st[k]:>8.3f}" for k in keys)
                             + f"{st['lag_brain_s']:>7.1f}{f['chi2']:>7.1f}")

    lines.append(f"\nRange of outcomes consistent with the paper (brain layer, lag-adjusted; "
                 f"power and identification from {studies} simulated 20-session studies)")
    lines.append(f"{'protocol':<9}{'w':>5}{'gap min-max':>15}{'power min-max':>15}"
                 f"{'ident min-max':>15}{'resid gap':>15}{'brain>scalp gap':>17}")
    for f in fits:
        lo, hi = f["gap_bounds"]["min"], f["gap_bounds"]["max"]
        a, b = lo["prediction"], hi["prediction"]
        lines.append(
            f"{f['protocol']:<9}{f['mix']:>5.2f}"
            f"{a['gap']:>8.3f}-{b['gap']:<6.3f}{a['power_gap']:>8.2f}-{b['power_gap']:<6.2f}"
            f"{a['identified']:>8.0f}-{b['identified']:<6.0f}"
            f"{a['resid_gap']:>8.3f}-{b['resid_gap']:<6.3f}"
            f"{a['share_brain_gt_scalp_gap']:>10.2f}-{b['share_brain_gt_scalp_gap']:<6.2f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
