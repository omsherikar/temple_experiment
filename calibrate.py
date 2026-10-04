#!/usr/bin/env python3
"""How often the test reports a gap for a device that only follows the maneuver.

    python calibrate.py --studies 100 [--gap 5,30]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial

import numpy as np

from mixed_pairs import run
from simulate import load_fitted_params, simulate_dataset


def one_study(seed: int, gap_s: tuple[float, float] | None = None) -> dict:
    params = load_fitted_params(0.0)
    if gap_s:
        params = {k: replace(p, gap_s=gap_s) for k, p in params.items()}
    res = run(simulate_dataset(0.0, params, seed=seed), "temple_bf", ["brain_hbo", "scalp_hbo"],
              n_perm=500, seed=seed)
    out = {}
    for protocol, r in res.items():
        for variant in ("full", "residual"):
            for kind in ("lag", "zero"):
                s = r["layers"][(variant, "brain_hbo", kind)]
                out[(protocol, variant, kind)] = (
                    s["p_gap_wilcoxon"], s["p_pairing_permutation"], s["p_identification"]
                )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="False-positive rate of the mixed-pair test.")
    ap.add_argument("--studies", type=int, default=100)
    ap.add_argument("--first-seed", type=int, default=100)
    ap.add_argument("--gap", help="transition gap range in s, e.g. 5,30")
    args = ap.parse_args()
    gap_s = tuple(float(g) for g in args.gap.split(",")) if args.gap else None

    seeds = range(args.first_seed, args.first_seed + args.studies)
    with ProcessPoolExecutor() as ex:
        studies = list(ex.map(partial(one_study, gap_s=gap_s), seeds))

    print(f"Share of {len(studies)} null studies with p < 0.05 (brain layer)")
    print(f"{'protocol':<9}{'variant':<10}{'kind':<6}{'p(gap)':>8}{'p(perm)':>9}{'p(id)':>7}")
    for key in studies[0]:
        rate = (np.array([s[key] for s in studies]) < 0.05).mean(axis=0)
        print(f"{key[0]:<9}{key[1]:<10}{key[2]:<6}{rate[0]:>8.2f}{rate[1]:>9.2f}{rate[2]:>7.2f}")


if __name__ == "__main__":
    main()
