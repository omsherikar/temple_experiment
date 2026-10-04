"""Screen candidate head-down-tilt mechanisms: global fit of each, chi2 vs the paper.

Run from the repository root: python results/hdt_screen.py
"""
import json, math, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fit_to_paper import PAPER_TARGETS, chi2, statistics_over_seeds, _de
from simulate import default_params

SE = next(f["se"] for f in json.load(open(Path(__file__).resolve().parents[1] / "paper_fit.json"))["fits"] if f["protocol"] == "hdt")
T = PAPER_TARGETS["hdt"]
BASE = [("brain_noise", math.log(0.01), math.log(3.0), True), ("bf_noise", math.log(0.01), math.log(3.0), True),
        ("bf_lag_s", 0.0, 30.0, False), ("bf_drift", -3.0, 3.0, False)]
VARIANTS = {
    "kinetics (A+B)": [("bf_tau_s", 0.0, 30.0, False), ("brain_fast_share", 0.0, 1.0, False)],
    "brain adaptation": [("brain_adapt", 0.0, 0.9, False), ("brain_adapt_tau_s", 10.0, 200.0, False)],
    "BF transition transient": [("bf_transient", -2.0, 2.0, False)],
    "adaptation + transient": [("brain_adapt", 0.0, 0.9, False), ("brain_adapt_tau_s", 10.0, 200.0, False),
                               ("bf_transient", -2.0, 2.0, False)],
    # Same, with the brain shape bounded by Fig. 3A (fast rise, <= 25% decline).
    "Fig.3A-bounded + transient": [("brain_fast_share", 0.0, 1.0, False), ("brain_adapt", 0.0, 0.25, False),
                                   ("brain_adapt_tau_s", 10.0, 200.0, False), ("bf_transient", -2.0, 2.0, False)],
}
if len(sys.argv) > 1:  # screen only the named variants
    VARIANTS = {k: v for k, v in VARIANTS.items() if k in sys.argv[1:]}

def build(spec, x):
    kw = {n: float(np.exp(v)) if lg else float(v) for (n, _, _, lg), v in zip(spec, x)}
    return replace(default_params()["hdt"], **kw)

def job(arg):
    name, mix = arg
    spec = BASE + VARIANTS[name]
    obj = lambda x: chi2(statistics_over_seeds(build(spec, x), "hdt", mix), T, SE)
    res = _de(obj, [(lo, hi) for _, lo, hi, _ in spec], maxiter=30)
    p = build(spec, res.x)
    st = statistics_over_seeds(p, "hdt", mix)
    return name, mix, float(res.fun), st, {n: round(getattr(p, n), 3) for n, *_ in spec}

if __name__ == "__main__":
    jobs = [(n, m) for n in VARIANTS for m in (0.0, 1.0)]
    with ProcessPoolExecutor() as ex:
        out = list(ex.map(job, jobs))
    print("paper  lag %.3f zero %.3f trans %.3f grp %.3f lag 13-15" % (T["r_lag_brain"], T["r_zero_brain"], T["r_transition"], T["r_group_brain"]))
    for name, mix, c, st, p in out:
        print(f"{name:<26} w={mix:.0f} chi2 {c:7.1f} | lag {st['r_lag_brain']:.3f} zero {st['r_zero_brain']:.3f} "
              f"trans {st['r_transition']:.3f} grp {st['r_group_brain']:.3f} lag_s {st['lag_brain_s']:.1f} | {p}")
    json.dump(out, open(Path(__file__).with_name("hdt_screen.json" if len(sys.argv) == 1 else "hdt_screen_bounded.json"), "w"), indent=1)
