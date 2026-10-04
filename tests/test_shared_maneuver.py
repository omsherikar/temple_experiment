from dataclasses import replace

import numpy as np
import pytest

import shared_maneuver_test as smt
from simulate import band_noise, default_params, simulate_dataset, write_dataset


def test_lagged_r_finds_known_lag():
    rng = np.random.default_rng(1)
    nirs = band_noise(600, rng, (0.01, 0.12))
    bf = np.concatenate([np.zeros(13), nirs[:-13]])  # BF follows NIRS by 13 s
    r, lag, r0 = smt.lagged_r(bf, nirs)
    assert lag == 13
    assert r > 0.99
    assert r0 < r


def test_cross_matrix_matches_pairwise_definition():
    rng = np.random.default_rng(2)
    bf = np.array([band_noise(300, rng, (0.01, 0.12)) for _ in range(4)])
    nirs = np.array([band_noise(300, rng, (0.01, 0.12)) for _ in range(4)])
    r_lag, r_zero = smt.cross_matrix(bf, nirs)
    for i in range(4):
        for j in range(4):
            r, _, r0 = smt.lagged_r(bf[i], nirs[j])
            assert r_lag[i, j] == pytest.approx(r)
            assert r_zero[i, j] == pytest.approx(r0)


def test_ema_filtfilt_matches_loop_definition():
    x = np.random.default_rng(0).standard_normal(50)
    a = smt.EMA_ALPHA

    def ema(v):
        out = [v[0]]
        for xi in v[1:]:
            out.append(a * xi + (1 - a) * out[-1])
        return np.array(out)

    assert smt.ema_filtfilt(x) == pytest.approx(ema(ema(x)[::-1])[::-1])


def test_ema_filtfilt_is_zero_phase():
    x = np.zeros(101)
    x[50] = 1.0
    y = smt.ema_filtfilt(x)
    assert int(np.argmax(y)) == 50
    assert y[45] == pytest.approx(y[55], rel=0.2)


def test_derangements_respect_subject_mask():
    subjects = ["a", "a", "b", "c", "d", "e"]
    mask = smt.eligible_mask(subjects)
    perms = smt.random_derangements(mask, 200, np.random.default_rng(0))
    n = len(subjects)
    assert mask[np.arange(n)[None, :], perms].all()
    assert not np.any(perms == np.arange(n))
    assert not np.any((perms[:, 0] == 1) | (perms[:, 1] == 0))  # same subject never paired


def test_mixed_pair_stats_on_constructed_matrix():
    n = 10
    R = np.full((n, n), 0.8)
    np.fill_diagonal(R, 0.95)
    mask = smt.eligible_mask([str(i) for i in range(n)])
    perms = smt.random_derangements(mask, 500, np.random.default_rng(0))
    s = smt.mixed_pair_stats(R, mask, perms)
    assert s["median_real"] == pytest.approx(0.95)
    assert s["median_mixed"] == pytest.approx(0.8)
    assert s["median_gap"] == pytest.approx(0.15)
    assert s["identified"] == n
    assert s["p_gap_wilcoxon"] < 0.01
    assert s["p_pairing_permutation"] < 0.01


def test_maneuver_design_absorbs_block_locked_response():
    events = np.array([0, 180, 195, 375, 390, 570, 585, 765, 780, 960], float)
    t = np.arange(0, 961, 1.0)
    X = smt.maneuver_design(t, events, (2, 4))
    task = X[:, 2]
    assert task[100] == 0 and task[300] == 1 and task[187] == pytest.approx(7 / 15)
    # A slow response delayed by 25 s is not exactly in the basis but is explained.
    resp = np.concatenate([np.zeros(25), smt.lowpass1(task, 30.0)[:-25]])
    coef, *_ = np.linalg.lstsq(X, resp, rcond=None)
    resid = resp - X @ coef
    assert resid.std() < 0.01 * resp.std()


@pytest.fixture(scope="module")
def tilt_and_person():
    kw = dict(n_perm=500, seed=0, residual=True)
    layers = ["brain_hbo", "scalp_hbo"]
    # Fixed test parameters, independent of paper_fit.json.
    params = {"hdt": replace(default_params()["hdt"], brain_noise=0.4, phys_share=0.8, bf_noise=0.15)}
    run = lambda mix: smt.run(  # noqa: E731
        simulate_dataset(mix, params, n_sessions=12, seed=3, protocols=("hdt",)), "temple_bf", layers, **kw
    )
    tilt, person = run(0.0), run(1.0)
    return tilt["hdt"], person["hdt"]


def test_both_devices_have_impressive_headline(tilt_and_person):
    for res in tilt_and_person:
        assert res["layers"][("full", "brain_hbo", "lag")]["median_real"] > 0.8


def test_only_person_device_shows_a_gap(tilt_and_person):
    tilt, person = tilt_and_person
    t = tilt["layers"][("full", "brain_hbo", "lag")]
    p = person["layers"][("full", "brain_hbo", "lag")]
    assert abs(t["median_gap"]) < 0.02
    assert p["median_gap"] > 0.04
    assert p["p_gap_wilcoxon"] < 0.001
    assert p["identified"] >= 10
    assert person["layers"][("residual", "brain_hbo", "lag")]["median_gap"] > 0.3
    assert abs(tilt["layers"][("residual", "brain_hbo", "lag")]["median_gap"]) < 0.1


def test_cli_end_to_end(tmp_path, capsys):
    write_dataset(simulate_dataset(1.0, default_params(), n_sessions=5, seed=7, protocols=("squat",)), tmp_path / "data")
    rc = smt.main([str(tmp_path / "data" / "manifest.csv"), "--perms", "200", "--out", str(tmp_path / "out")])
    assert rc == 0
    text = capsys.readouterr().out
    assert "squat" in text and "brain_hbo" in text
    assert (tmp_path / "out" / "per_session.csv").exists()
    assert (tmp_path / "out" / "summary.json").exists()
    assert (tmp_path / "out" / "matrix_squat_full_brain_hbo_lag.csv").exists()


def test_paper_statistics_recover_noise_free_truth():
    from fit_to_paper import paper_statistics

    p = replace(default_params()["squat"], brain_noise=0.001, bf_noise=0.001, white_noise=0.0, bf_lag_s=14.0)
    st = paper_statistics(simulate_dataset(1.0, {"squat": p}, n_sessions=8, seed=1, protocols=("squat",)), "squat")
    assert st["lag_brain_s"] == pytest.approx(14, abs=1)
    assert st["r_lag_brain"] > 0.99
    assert st["r_transition"] > 0.99  # lag-aligned windows
    assert st["r_group_brain"] > 0.99
    assert st["r_zero_brain"] < st["r_lag_brain"]
