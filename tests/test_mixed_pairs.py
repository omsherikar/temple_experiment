from dataclasses import replace

import numpy as np
import pytest

import mixed_pairs as mp
from fit import paper_statistics
from simulate import band_noise, default_params, simulate_dataset, write_dataset


def test_lagged_r_finds_known_lag():
    nirs = band_noise(600, np.random.default_rng(1), (0.01, 0.12))
    bf = np.concatenate([np.zeros(13), nirs[:-13]])
    r, lag, r0 = mp.lagged_r(bf, nirs)
    assert lag == 13
    assert r > 0.99
    assert r0 < r


def test_cross_matrix_matches_lagged_r():
    rng = np.random.default_rng(2)
    bf = np.array([band_noise(300, rng, (0.01, 0.12)) for _ in range(4)])
    nirs = np.array([band_noise(300, rng, (0.01, 0.12)) for _ in range(4)])
    r_lag, r_zero = mp.cross_matrix(bf, nirs)
    for i in range(4):
        for j in range(4):
            r, _, r0 = mp.lagged_r(bf[i], nirs[j])
            assert r_lag[i, j] == pytest.approx(r)
            assert r_zero[i, j] == pytest.approx(r0)


def test_ema_filtfilt_matches_loop():
    x = np.random.default_rng(0).standard_normal(50)
    a = mp.EMA_ALPHA

    def ema(v):
        out = [v[0]]
        for xi in v[1:]:
            out.append(a * xi + (1 - a) * out[-1])
        return np.array(out)

    assert mp.ema_filtfilt(x) == pytest.approx(ema(ema(x)[::-1])[::-1])


def test_ema_filtfilt_is_zero_phase():
    x = np.zeros(101)
    x[50] = 1.0
    y = mp.ema_filtfilt(x)
    assert int(np.argmax(y)) == 50
    assert y[45] == pytest.approx(y[55], rel=0.2)


def test_random_pairings_avoid_same_subject():
    subjects = ["a", "a", "b", "c", "d", "e"]
    mask = mp.mixed_pair_mask(subjects)
    perms = mp.random_pairings(mask, 200, np.random.default_rng(0))
    n = len(subjects)
    assert mask[np.arange(n)[None, :], perms].all()
    assert not np.any(perms == np.arange(n))
    assert not np.any((perms[:, 0] == 1) | (perms[:, 1] == 0))


def test_mixed_pair_stats():
    n = 10
    R = np.full((n, n), 0.8)
    np.fill_diagonal(R, 0.95)
    mask = mp.mixed_pair_mask([str(i) for i in range(n)])
    s = mp.mixed_pair_stats(R, mask, mp.random_pairings(mask, 500, np.random.default_rng(0)))
    assert s["median_real"] == pytest.approx(0.95)
    assert s["median_mixed"] == pytest.approx(0.8)
    assert s["median_gap"] == pytest.approx(0.15)
    assert s["identified"] == n
    assert s["p_gap_wilcoxon"] < 0.01
    assert s["p_pairing_permutation"] < 0.01


def test_block_regressors_explain_delayed_response():
    events = np.array([0, 180, 195, 375, 390, 570, 585, 765, 780, 960], float)
    t = np.arange(0, 961, 1.0)
    X = mp.block_regressors(t, events, (2, 4))
    task = X[:, 2]
    assert task[100] == 0 and task[300] == 1 and task[187] == pytest.approx(7 / 15)
    resp = np.concatenate([np.zeros(25), mp.lowpass1(task, 30.0)[:-25]])
    coef, *_ = np.linalg.lstsq(X, resp, rcond=None)
    assert (resp - X @ coef).std() < 0.01 * resp.std()


@pytest.fixture(scope="module")
def tilt_and_person():
    params = {"hdt": replace(default_params()["hdt"], brain_noise=0.4, phys_share=0.8, bf_noise=0.15)}

    def run(mix):
        sessions = simulate_dataset(mix, params, n_sessions=12, seed=3, protocols=("hdt",))
        return mp.run(sessions, "temple_bf", ["brain_hbo", "scalp_hbo"], n_perm=500, seed=0)["hdt"]

    return run(0.0), run(1.0)


def test_both_devices_have_high_headline_r(tilt_and_person):
    for res in tilt_and_person:
        assert res["layers"][("full", "brain_hbo", "lag")]["median_real"] > 0.8


def test_only_tracking_device_shows_a_gap(tilt_and_person):
    tilt, person = tilt_and_person
    t = tilt["layers"][("full", "brain_hbo", "lag")]
    p = person["layers"][("full", "brain_hbo", "lag")]
    assert abs(t["median_gap"]) < 0.02
    assert p["median_gap"] > 0.04
    assert p["p_gap_wilcoxon"] < 0.001
    assert p["identified"] >= 10
    assert person["layers"][("residual", "brain_hbo", "lag")]["median_gap"] > 0.3
    assert abs(tilt["layers"][("residual", "brain_hbo", "lag")]["median_gap"]) < 0.1


def test_cli(tmp_path, capsys):
    sessions = simulate_dataset(1.0, default_params(), n_sessions=5, seed=7, protocols=("squat",))
    write_dataset(sessions, tmp_path / "data")
    assert mp.main([str(tmp_path / "data" / "manifest.csv"), "--perms", "200", "--out", str(tmp_path / "out")]) == 0
    text = capsys.readouterr().out
    assert "squat" in text and "brain_hbo" in text
    for name in ("per_session.csv", "summary.json", "matrix_squat_full_brain_hbo_lag.csv"):
        assert (tmp_path / "out" / name).exists()


def test_paper_statistics_without_noise():
    p = replace(default_params()["squat"], brain_noise=0.001, bf_noise=0.001, white_noise=0.0, bf_lag_s=14.0)
    st = paper_statistics(simulate_dataset(1.0, {"squat": p}, n_sessions=8, seed=1, protocols=("squat",)), "squat")
    assert st["lag_brain_s"] == pytest.approx(14, abs=1)
    assert st["r_lag_brain"] > 0.99
    assert st["r_transition"] > 0.99
    assert st["r_group_brain"] > 0.99
    assert st["r_zero_brain"] < st["r_lag_brain"]


def _noise_free(**kw):
    p = replace(default_params()["hdt"], brain_noise=0.0, bf_noise=0.0, white_noise=0.0, **kw)
    return simulate_dataset(0.0, {"hdt": p}, n_sessions=1, seed=0, protocols=("hdt",))[0]


def test_head_down_tilt_mechanisms():
    shape = dict(brain_fast_share=1.0, brain_adapt=0.2, brain_adapt_tau_s=30.0, bf_lag_s=14.0)
    s = _noise_free(bf_transient=0.8, **shape)
    ev, brain = s.events, s.signals["brain_hbo"]
    onset = int(ev[2])
    assert brain[onset + 5] > 0.8 * brain[onset : onset + 30].max()
    assert brain[int(ev[3]) - 1] == pytest.approx(0.8, abs=0.05)

    transient = s.signals["temple_bf"] - _noise_free(**shape).signals["temple_bf"]
    assert transient[int(ev[1]) : onset + 10].max() > 0.5
    assert abs(transient[int((ev[2] + ev[3]) / 2)]) < 0.01
