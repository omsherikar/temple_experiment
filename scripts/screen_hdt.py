#!/usr/bin/env python3
"""Fit candidate head-down tilt mechanisms and compare their chi-square.

    python scripts/screen_hdt.py [variant ...]
"""

from __future__ import annotations

import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fit import PAPER_TARGETS, _de, chi2, statistics_over_seeds  # noqa: E402
from simulate import default_params  # noqa: E402

TARGETS = PAPER_TARGETS["hdt"]
SE = next(f["se"] for f in json.loads((ROOT / "fit.json").read_text())["fits"] if f["protocol"] == "hdt")

BASE = [
    ("brain_noise", math.log(0.01), math.log(3.0), True),
    ("bf_noise", math.log(0.01), math.log(3.0), True),
    ("bf_lag_s", 0.0, 30.0, False),
    ("bf_drift", -3.0, 3.0, False),
]
VARIANTS = {
    "kinetics (A+B)": [("bf_tau_s", 0.0, 30.0, False), ("brain_fast_share", 0.0, 1.0, False)],
    "brain adaptation": [("brain_adapt", 0.0, 0.9, False), ("brain_adapt_tau_s", 10.0, 200.0, False)],
    "BF transition transient": [("bf_transient", -2.0, 2.0, False)],
    "adaptation + transient": [
        ("brain_adapt", 0.0, 0.9, False), ("brain_adapt_tau_s", 10.0, 200.0, False),
        ("bf_transient", -2.0, 2.0, False),
    ],
    "Fig.3A-bounded + transient": [
        ("brain_fast_share", 0.0, 1.0, False), ("brain_adapt", 0.0, 0.25, False),
        ("brain_adapt_tau_s", 10.0, 200.0, False), ("bf_transient", -2.0, 2.0, False),
    ],
}


def build(spec, x):
    return replace(default_params()["hdt"],
                   **{n: float(np.exp(v)) if lg else float(v) for (n, _, _, lg), v in zip(spec, x)})


def screen(arg):
    name, mix = arg
    spec = BASE + VARIANTS[name]
    res = _de(lambda x: chi2(statistics_over_seeds(build(spec, x), "hdt", mix), TARGETS, SE),
              [(lo, hi) for _, lo, hi, _ in spec], maxiter=30)
    p = build(spec, res.x)
    return name, mix, float(res.fun), statistics_over_seeds(p, "hdt", mix), {n: round(getattr(p, n), 3) for n, *_ in spec}


def main() -> None:
    names = [n for n in VARIANTS if n in sys.argv[1:]] or list(VARIANTS)
    with ProcessPoolExecutor() as ex:
        out = list(ex.map(screen, [(n, m) for n in names for m in (0.0, 1.0)]))
    print("paper  lag %.3f zero %.3f trans %.3f grp %.3f lag 13-15" % (
        TARGETS["r_lag_brain"], TARGETS["r_zero_brain"], TARGETS["r_transition"], TARGETS["r_group_brain"]))
    for name, mix, c, st, p in out:
        print(f"{name:<26} w={mix:.0f} chi2 {c:7.1f} | lag {st['r_lag_brain']:.3f} zero {st['r_zero_brain']:.3f} "
              f"trans {st['r_transition']:.3f} grp {st['r_group_brain']:.3f} lag_s {st['lag_brain_s']:.1f} | {p}")
    suffix = "" if len(names) == len(VARIANTS) else "_bounded"
    (ROOT / "results" / f"screen_hdt{suffix}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
