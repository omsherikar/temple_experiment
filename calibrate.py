#!/usr/bin/env python3
"""False-positive check: how often does the test report a gap when there is none?

Simulates many independent 20-session studies with the "tilt" device (it
follows the imposed maneuver, in each session's own timing, and nothing
else) and counts how often each p value falls below 0.05. A well calibrated
test should do that about 5% of the time.

    python calibrate.py --studies 100
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from shared_maneuver_test import run
from simulate import simulate_dataset

LAYERS = ["brain_hbo", "scalp_hbo"]


def one_study(seed: int) -> dict:
    res = run(simulate_dataset(0.0, seed=seed), "temple_bf", LAYERS, n_perm=500, seed=seed)
    out = {}
    for protocol, r in res.items():
        for variant in ("full", "residual"):
            for kind in ("lag", "zero"):
                s = r["layers"][(variant, "brain_hbo", kind)]
                out[(protocol, variant, kind)] = (
                    s["p_gap_wilcoxon"],
                    s["p_pairing_permutation"],
                    s["p_identification"],
                )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--studies", type=int, default=100)
    ap.add_argument("--first-seed", type=int, default=100)
    args = ap.parse_args()

    seeds = range(args.first_seed, args.first_seed + args.studies)
    with ProcessPoolExecutor() as ex:
        studies = list(ex.map(one_study, seeds))

    print(f"Share of {len(studies)} simulated null studies with p < 0.05 (brain layer, tilt-only device)")
    print(f"{'protocol':<9}{'variant':<10}{'kind':<6}{'p(gap)':>8}{'p(perm)':>9}{'p(id)':>7}")
    for key in studies[0]:
        rate = (np.array([s[key] for s in studies]) < 0.05).mean(axis=0)
        print(f"{key[0]:<9}{key[1]:<10}{key[2]:<6}{rate[0]:>8.2f}{rate[1]:>9.2f}{rate[2]:>7.2f}")


if __name__ == "__main__":
    main()
