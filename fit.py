#!/usr/bin/env python3
"""Fit the simulator to the statistics published by Gulati et al. 2026.

For each tracking share (mix) the free parameters are fitted to the paper's
Table 1, Table 2 and Fig. 4 values. Agreement is a chi-square in the paper's
sampling error (5 df, consistent if <= 11.07). Among consistent parameter
sets the script then finds the smallest and largest mixed-pair gap and the
power of the test at both ends.

    python fit.py --protocols squat,supine
    python fit.py --protocols hdt --search-effort 2 --out results/fit_hdt.json
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

from mixed_pairs import _unit_rows, align_sessions, canonical_events, cross_matrix, fisher_mean, lagged_r, run
from simulate import FIT_FILE, SimParams, default_params, simulate_dataset

PAPER_TARGETS = {  # Table 1 (lag, zero), Table 2 (transition), Fig. 4 (group)
    "hdt": {"r_lag_brain": 0.912, "r_lag_scalp": 0.494, "r_zero_brain": 0.826,
            "r_transition": 0.878, "r_group_brain": 0.915, "r_group_scalp": 0.430},
    "squat": {"r_lag_brain": 0.843, "r_lag_scalp": 0.700, "r_zero_brain": 0.701,
              "r_transition": 0.861, "r_group_brain": 0.916, "r_group_scalp": 0.783},
    "supine": {"r_lag_brain": 0.839, "r_lag_scalp": 0.767, "r_zero_brain": 0.796,
               "r_transition": 0.876, "r_group_brain": 0.941, "r_group_scalp": 0.929},
}
TRANSITION_HALF_WINDOW_S = {"hdt": 90, "squat": 45, "supine": 90}  # Sec. 2.6
TARGET_LAG_S = 14.0  # Sec. 3.2: median lag 13-15 s

FIT_SEEDS = (1000, 1001, 1002, 1003)
SE_SEEDS = tuple(range(2000, 2030))
CHI2_95_DF5 = 11.07
BRAIN_KEYS = ("r_lag_brain", "r_zero_brain", "r_transition", "r_group_brain")
SCALP_KEYS = ("r_lag_scalp", "r_group_scalp")

# (name, lower, upper, log scale)
BRAIN_SPEC = [
    ("brain_noise", 0.01, 3.0, True),
    ("bf_noise", 0.01, 3.0, True),
    ("bf_lag_s", 0.0, 30.0, False),
    ("bf_drift", -3.0, 3.0, False),
]
# The base model cannot reproduce head-down tilt (see scripts/screen_hdt.py).
# Brain-shape bounds come from Fig. 3A: near-instant rise, ~12% decline.
EXTRA_SPEC = {
    "hdt": [
        ("brain_fast_share", 0.0, 1.0, False),
        ("brain_adapt", 0.0, 0.25, False),
        ("brain_adapt_tau_s", 10.0, 200.0, False),
        ("bf_transient", -2.0, 2.0, False),
    ],
}
SCALP_BOUNDS = [(-5.0, 5.0), (math.log(1.0), math.log(200.0))]

search_effort = 1.0


def paired_lagged_r(bf: np.ndarray, nirs: np.ndarray, max_lag: int = 30):
    """Row-wise (max r over lags, its lag, zero-lag r)."""
    m = bf.shape[1]
    best = np.full(bf.shape[0], -np.inf)
    best_lag = np.zeros(bf.shape[0], int)
    r0 = None
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            r = np.sum(_unit_rows(bf[:, lag:]) * _unit_rows(nirs[:, : m - lag]), axis=1)
        else:
            r = np.sum(_unit_rows(bf[:, : m + lag]) * _unit_rows(nirs[:, -lag:]), axis=1)
        better = r > best
        best[better], best_lag[better] = r[better], lag
        if lag == 0:
            r0 = r
    return best, best_lag, r0


def paper_statistics(sessions, protocol: str) -> dict[str, float]:
    data = align_sessions(sessions, "warp", smooth=True)
    canon = canonical_events(sessions)
    bf, brain, scalp = data["temple_bf"], data["brain_hbo"], data["scalp_hbo"]

    r_lag_b, lag_b, r0_b = paired_lagged_r(bf, brain)
    r_lag_s, _, _ = paired_lagged_r(bf, scalp)

    # Transition windows are lag-aligned: with a 14 s lag, a zero-lag window
    # could not reach the reported squat value (0.861) at the paper's noise.
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

    R, _ = cross_matrix(bf, brain)
    n = R.shape[0]
    off = ~np.eye(n, dtype=bool)
    mixed = np.array([fisher_mean(np.concatenate([R[i, off[i]], R[off[:, i], i]])) for i in range(n)])

    return {
        "r_lag_brain": float(np.median(r_lag_b)),
        "r_lag_scalp": float(np.median(r_lag_s)),
        "r_zero_brain": float(np.median(r0_b)),
        "r_transition": float(np.median(r_trans)),
        "r_group_brain": lagged_r(bf.mean(0), brain.mean(0))[0],
        "r_group_scalp": lagged_r(bf.mean(0), scalp.mean(0))[0],
        "lag_brain_s": float(np.median(lag_b)),
        "gap": float(np.median(np.diag(R) - mixed)),
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
    err = sum((_z(stats[k]) - _z(targets[k])) ** 2 for k in keys)
    if "r_lag_brain" in keys:
        err += ((stats["lag_brain_s"] - TARGET_LAG_S) / 5.0) ** 2
    return err


def standard_errors(params: SimParams, protocol: str, mix: float) -> dict[str, float]:
    per = [paper_statistics(simulate_dataset(mix, {protocol: params}, seed=s, protocols=(protocol,)), protocol)
           for s in SE_SEEDS]
    se = {k: float(np.std([_z(d[k]) for d in per], ddof=1)) for k in BRAIN_KEYS + SCALP_KEYS}
    # The paper only gives the lag as "13 to 15 s".
    se["lag_brain_s"] = max(float(np.std([d["lag_brain_s"] for d in per], ddof=1)), 1.0)
    return se


def chi2(stats: dict[str, float], targets: dict[str, float], se: dict[str, float]) -> float:
    err = sum(((_z(stats[k]) - _z(targets[k])) / se[k]) ** 2 for k in BRAIN_KEYS)
    return err + ((stats["lag_brain_s"] - TARGET_LAG_S) / se["lag_brain_s"]) ** 2


def _set_effort(effort: float) -> None:
    global search_effort
    search_effort = effort


def _de(objective, bounds, x0=None, seed=0, maxiter=30):
    # Gradient-free global search: the objective is built from medians and
    # lag maxima, so it is not smooth.
    return optimize.differential_evolution(
        objective, bounds, x0=x0, seed=seed, popsize=8, maxiter=int(maxiter * search_effort),
        tol=1e-4, polish=False,
    )


def _spec(protocol: str) -> list:
    return BRAIN_SPEC + EXTRA_SPEC.get(protocol, [])


def _bounds(spec) -> list:
    return [(math.log(lo), math.log(hi)) if lg else (lo, hi) for _, lo, hi, lg in spec]


def _with(base: SimParams, x, spec) -> SimParams:
    return replace(base, **{n: float(np.exp(v)) if lg else float(v) for (n, _, _, lg), v in zip(spec, x)})


def _as_x(p: SimParams, spec) -> np.ndarray:
    return np.array([math.log(getattr(p, n)) if lg else getattr(p, n) for n, _, _, lg in spec])


def _with_scalp(p: SimParams, x) -> SimParams:
    return replace(p, scalp_amp=float(x[0]), scalp_tau_s=float(np.exp(x[1])))


def fit_one(job: tuple[str, float, SimParams, bool]) -> dict:
    protocol, mix, base, with_bounds = job
    targets = PAPER_TARGETS[protocol]
    spec = _spec(protocol)
    bounds = _bounds(spec)

    def stats(p):
        return statistics_over_seeds(p, protocol, mix)

    p0 = _with(base, _de(lambda x: raw_loss(stats(_with(base, x, spec)), targets, BRAIN_KEYS), bounds).x, spec)
    se = standard_errors(p0, protocol, mix)
    best = _de(lambda x: chi2(stats(_with(base, x, spec)), targets, se), bounds, x0=_as_x(p0, spec), maxiter=20)
    chi2_min = float(best.fun)

    scalp = _de(lambda x: raw_loss(stats(_with_scalp(_with(base, best.x, spec), x)), targets, SCALP_KEYS),
                SCALP_BOUNDS)
    p1 = _with_scalp(_with(base, best.x, spec), scalp.x)

    limit = max(CHI2_95_DF5, chi2_min)
    ends = {}
    for name, sign in (("min", 1.0), ("max", -1.0)):
        if not with_bounds:
            ends[name] = p1
            continue

        def bound_obj(x, sign=sign):
            st = stats(_with(base, x, spec))
            return sign * st["gap"] + 100.0 * max(0.0, chi2(st, targets, se) - limit)

        res = _de(bound_obj, bounds, x0=_as_x(p1, spec), maxiter=25)
        ends[name] = _with_scalp(_with(base, res.x, spec), scalp.x)

    out = {
        "protocol": protocol,
        "mix": mix,
        "params": asdict(p1),
        "chi2": chi2_min,
        "consistent": chi2_min <= CHI2_95_DF5,
        "se": se,
        "fitted_stats": stats(p1),
        "scalp_loss": float(scalp.fun),
        "targets": targets,
        "gap_bounds": {},
    }
    for name, p in ends.items():
        st = stats(p)
        out["gap_bounds"][name] = {"params": asdict(p), "stats": st, "chi2": chi2(st, targets, se)}
    return out


def predict_one(job: tuple[str, float, dict, int]) -> dict:
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


def report(fits: list[dict], studies: int) -> str:
    keys = list(PAPER_TARGETS["hdt"])
    short = ["lag b", "lag s", "zero b", "trans", "grp b", "grp s"]
    lines = ["Best fit (paper in the first row of each block); consistent if chi2 <= 11.07"]
    for protocol in dict.fromkeys(f["protocol"] for f in fits):
        t = PAPER_TARGETS[protocol]
        lines.append(f"\n{protocol:<8}{'w':>5}" + "".join(f"{s:>8}" for s in short) + f"{'lag s':>7}{'chi2':>7}")
        lines.append(f"{'paper':<8}{'':>5}" + "".join(f"{t[k]:>8.3f}" for k in keys) + f"{'13-15':>7}")
        for f in fits:
            if f["protocol"] == protocol:
                st = f["fitted_stats"]
                lines.append(f"{'':<8}{f['mix']:>5.2f}" + "".join(f"{st[k]:>8.3f}" for k in keys)
                             + f"{st['lag_brain_s']:>7.1f}{f['chi2']:>7.1f}")

    lines.append(f"\nOutcomes consistent with the paper (brain layer, {studies} studies of 20 sessions)")
    lines.append(f"{'protocol':<9}{'w':>5}{'gap min-max':>15}{'power min-max':>15}"
                 f"{'ident min-max':>15}{'resid gap':>15}{'brain>scalp gap':>17}")
    for f in fits:
        a, b = f["gap_bounds"]["min"]["prediction"], f["gap_bounds"]["max"]["prediction"]
        lines.append(
            f"{f['protocol']:<9}{f['mix']:>5.2f}"
            f"{a['gap']:>8.3f}-{b['gap']:<6.3f}{a['power_gap']:>8.2f}-{b['power_gap']:<6.2f}"
            f"{a['identified']:>8.0f}-{b['identified']:<6.0f}"
            f"{a['resid_gap']:>8.3f}-{b['resid_gap']:<6.3f}"
            f"{a['share_brain_gt_scalp_gap']:>10.2f}-{b['share_brain_gt_scalp_gap']:<6.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit the simulator to the paper's statistics.")
    ap.add_argument("--mixes", default="0,0.5,1")
    ap.add_argument("--protocols", default="hdt,squat,supine")
    ap.add_argument("--phys-share", type=float, default=0.5)
    ap.add_argument("--band-low", type=float, default=0.01, help="Hz")
    ap.add_argument("--gap", default="8,20", help="transition gap range, s")
    ap.add_argument("--studies", type=int, default=40)
    ap.add_argument("--search-effort", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=FIT_FILE)
    args = ap.parse_args()
    _set_effort(args.search_effort)

    gap = tuple(float(g) for g in args.gap.split(","))
    bases = {
        p: replace(b, phys_share=args.phys_share, band_hz=(args.band_low, b.band_hz[1]), gap_s=gap)
        for p, b in default_params().items()
    }
    # With mix = 0 the gap is zero by construction, so only the fit is needed.
    jobs = [(p, m, bases[p], m > 0) for p in args.protocols.split(",") for m in map(float, args.mixes.split(","))]
    with ProcessPoolExecutor(initializer=_set_effort, initargs=(search_effort,)) as ex:
        fits = list(ex.map(fit_one, jobs))
        preds = list(ex.map(predict_one, [(f["protocol"], f["mix"], f["gap_bounds"][e]["params"], args.studies)
                                          for f in fits for e in ("min", "max")]))
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


if __name__ == "__main__":
    main()
