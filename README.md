# Shared-maneuver test for the Temple-BF validation study

A small Python tool that answers one question about
[Gulati et al., *Evaluating Brain Flow Index from a temple-worn wearable against depth-resolved time-domain NIRS*](https://doi.org/10.64898/2026.09.17.751664)
(bioRxiv, 2026):

> How much of the Temple-BF / brain-layer correlation comes from everyone doing the
> same tilt at the same time, and how much is specific to the session?

The paper raises this itself. Limitation 2 says the common block design "can produce
high correlations when two sensors respond to the same imposed maneuver even if their
tissue sources differ." This tool puts a number on that effect. It is meant as a way to
measure that limitation, not as a challenge to the result.

## The idea

Every session in a protocol follows the same five blocks, in the same order, with the
same durations. So any two sessions share the maneuver, but only a session paired with
itself shares the person and the moment. The tool compares:

| | pairs | what they share |
|---|---|---|
| **real pairs** | Temple-BF of session *i* with NIRS of session *i* | maneuver + person + session |
| **mixed pairs** | Temple-BF of session *i* with NIRS of session *j* (different subject) | maneuver only |

**gap = real − mixed** is the part of the headline correlation that the shared maneuver
does not explain.

This is the standard "random pair" (pseudo-pair) control from fNIRS hyperscanning and
inter-subject correlation work, where random pairings are known to show high coherence
just from a shared task (Nguyen, Hoehl & Vrtička 2021).

## A hint that is already in the paper

Fig. 4 shows correlations between the *group-averaged* traces. Averaging across 20
sessions keeps only the response that everyone shares. Those correlations are as high as
the median per-session ones:

| protocol | Fig. 4, group-mean traces, r(brain) | Table 1, per-session median, lag-adjusted r(brain) |
|---|---|---|
| head-down tilt | 0.915 | 0.912 |
| stand-to-squat | 0.916 | 0.843 |
| stand-to-supine | 0.941 | 0.839 |

Averaging also removes noise, so this is not a proper test. But it suggests the shared
response alone could reach the headline numbers. The mixed-pair test measures this
properly, session by session.

## What the script computes

It uses the paper's own correlation method (Sec. 2.5–2.6), so real-pair numbers should
match the paper's:

- signals interpolated to a 1 Hz grid over the event-bounded interval
- forward-backward exponential moving average, α = 0.35
- Pearson r at every integer lag in ±30 s, keeping the largest (lag-adjusted r), plus
  the zero-lag r
- one-sided Wilcoxon signed-rank tests

Mixed pairs go through **exactly the same pipeline, including the ±30 s lag search**, so
the lag search gives them the same advantage it gives real pairs.

For each protocol and each NIRS layer (brain, scalp) it reports:

| output | meaning |
|---|---|
| `real` | median r of same-session pairs (the paper's headline) |
| `mixed` | median, over sessions, of the Fisher-averaged r of all different-subject pairs involving that session |
| `gap` | median per-session `real − mixed` |
| `p(gap)` | one-sided Wilcoxon signed-rank, gap > 0. **Primary test.** |
| `p(perm)` | permutation test: true pairing vs 10,000 random different-subject pairings (secondary, see calibration) |
| `ident` | how many sessions' Temple-BF correlates best with *its own* NIRS trace, out of all sessions (chance ≈ 1) |
| brain − scalp | the paper's brain-vs-scalp contrast, repeated for real pairs, mixed pairs, and the gap |

**Secondary analysis (residual).** Before correlating, each signal has everything its
own session's block timing can explain regressed out. The regressors are the task-block
indicator passed through 1st- and 2nd-order lags with time constants of 2–80 s, plus a
trend. What is left is activity that is not locked to the maneuver. This is the closest
this dataset gets to Limitation 6, sensitivity to "smaller spontaneous changes typical
of daily life".

**Alignment.** Mixed pairs need a shared protocol clock. By default (`--align warp`)
each segment between event markers is stretched onto the median timing, so block
onsets line up across sessions. This gives mixed pairs the best chance of matching,
which is the conservative choice. `--align onset` only aligns the first event.

**Repeat participants.** Pairs from the same `subject_id` are never used as mixed pairs.

## Results on simulated data

The data below are simulated, not real. Block timings follow the paper. Response shapes,
amplitudes and noise levels are my own assumptions, chosen so the headline r looks like
the paper's. There are two imaginary devices:

- **tilt**: follows a generic response to the maneuver (in that session's own timing)
  and never sees the session's own brain fluctuations
- **person**: follows that session's actual brain trace
- **person\***: the same "person" device in a world where every brain responds
  identically

20 sessions per protocol, lag-adjusted brain-layer r (`python simulate.py demo --perms 10000`):

| device | protocol | real | mixed | gap | p(gap) | ident | residual real | residual mixed |
|---|---|---|---|---|---|---|---|---|
| tilt | hdt | 0.902 | 0.892 | 0.004 | 0.205 | 1/20 | 0.152 | 0.174 |
| tilt | squat | 0.899 | 0.900 | −0.003 | 0.689 | 1/20 | 0.245 | 0.288 |
| tilt | supine | 0.895 | 0.894 | −0.004 | 0.861 | 2/20 | 0.183 | 0.180 |
| person | hdt | 0.969 | 0.899 | 0.070 | <0.001 | 20/20 | 0.697 | 0.149 |
| person | squat | 0.974 | 0.905 | 0.070 | <0.001 | 20/20 | 0.681 | 0.291 |
| person | supine | 0.967 | 0.891 | 0.071 | <0.001 | 20/20 | 0.714 | 0.171 |
| person\* | hdt | 0.969 | 0.969 | 0.001 | 0.123 | 4/20 | 0.138 | 0.134 |
| person\* | squat | 0.973 | 0.973 | 0.000 | 0.392 | 3/20 | 0.200 | 0.213 |
| person\* | supine | 0.969 | 0.969 | 0.000 | 0.608 | 2/20 | 0.125 | 0.133 |

What this shows:

1. **Both devices get an impressive headline.** In this setup the maneuver alone gives
   r ≈ 0.89–0.90. Only the gap and the identification count tell the devices apart.
2. **The gap grows with how much the device tracks the session.** Blending the two
   devices, head-down tilt:

   | share of "person" signal | 0 | 0.25 | 0.5 | 0.75 | 1 |
   |---|---|---|---|---|---|
   | real | 0.902 | 0.929 | 0.950 | 0.965 | 0.969 |
   | mixed | 0.892 | 0.905 | 0.911 | 0.909 | 0.899 |
   | gap | 0.004 | 0.020 | 0.035 | 0.055 | 0.070 |
   | p(gap) | 0.205 | <0.001 | <0.001 | <0.001 | <0.001 |
   | ident | 1/20 | 5/20 | 17/20 | 19/20 | 20/20 |

3. **The brain-vs-scalp contrast can come entirely from the shared maneuver.** For the
   "tilt" device, the brain-minus-scalp difference on real pairs is large and
   significant (hdt 0.92, squat 0.14, supine 0.08, all p < 0.001). But it is just as
   large on mixed pairs (1.03, 0.15, 0.09), and the difference in *gaps* is about zero.
   The brain layer's average response shape simply resembles the device's average
   response shape more than the scalp layer's does. That is still informative, but it
   is a statement about response shapes, not about tracking a person. The "person"
   device keeps a brain-over-scalp advantage in the gap for squat and supine
   (0.066 and 0.079, p < 0.001). So the tool reports the contrast at all three levels.
4. **A small gap is not a failure** (row `person*`). If everyone's brain responds the
   same way, even a perfect device shows no gap.

### Calibration

How often does the test report a gap when the device only follows the tilt? Here are
100 simulated null studies (`python calibrate.py --studies 100`), showing the share with
p < 0.05:

| protocol | analysis | p(gap) | p(perm) | p(id) |
|---|---|---|---|---|
| hdt | full, lag-adjusted | 0.02 | 0.09 | 0.02 |
| squat | full, lag-adjusted | 0.03 | 0.09 | 0.05 |
| supine | full, lag-adjusted | 0.01 | 0.05 | 0.03 |
| hdt | residual, lag-adjusted | 0.02 | 0.02 | 0.02 |
| squat | residual, lag-adjusted | 0.03 | 0.05 | 0.04 |
| supine | residual, lag-adjusted | 0.06 | 0.07 | 0.05 |

The primary test, `p(gap)`, stays at or below the nominal 5% (slightly above for
supine residual, 6%). On full sessions, the permutation test runs somewhat liberal
(up to 9–12% across variants). The cause is that real pairs share their session's exact
transition timing, which mixed pairs cannot fully reproduce even after warping. This is
why `p(gap)` is the primary test. The residual analysis removes timing-locked
components and is calibrated.

## How to read a result on the real recordings

- **A clear gap on the brain layer** means Temple-BF carries information about the
  individual session beyond the imposed maneuver. That strengthens the paper's claim.
  A larger gap for brain than for scalp would be a stronger version of the paper's
  brain-vs-scalp argument.
- **A small gap does not mean the device fails.** It can mean that people's brain
  responses to a tilt are very similar, so there is little session-specific signal to
  track (see `person*`).
- **Mixed / real** is a rough estimate of how much of the headline the maneuver alone
  explains.

## What it cannot show

- **Tissue source.** A gap shows session-specific shared information. It does not show
  that the information comes from the brain and not from blood pressure, CO₂ or skin
  perfusion, which drive both signals. That needs the systemic measurements the paper
  lists as future work.
- **Maneuver execution vs physiology.** A session-specific match can also come from how
  *that* session's maneuver was carried out: exact transition timing, how deep a squat
  was, small movements. Temple-BF uses accelerometer data, so it may see some of this.
  The residual analysis removes responses locked to the session's own block timing, but
  not things like squat depth.
- **Power.** With 20 sessions per protocol, small gaps may not reach significance.
- **Simulation.** Every number above comes from my assumptions about signal shapes and
  noise. They illustrate the method and are not predictions about the real data.

## Usage

```bash
pip install -r requirements.txt

python simulate.py demo                       # the tables above
python calibrate.py --studies 100             # false-positive check (~2 min on 4 cores)
python -m pytest                              # tests

# on recordings
python shared_maneuver_test.py manifest.csv --out results/
```

Options: `--layers brain_hbo,scalp_hbo` (NIRS columns; the first two are contrasted),
`--bf temple_bf`, `--align warp|onset`, `--max-lag 30`, `--no-smooth`,
`--no-residual`, `--task-blocks 2,4`, `--perms 10000`, `--seed 0`.

### Input format

`manifest.csv`, one row per session:

```csv
session_id,subject_id,protocol,file,events
s01,p01,hdt,sessions/s01.csv,0;180;195;375;390;570;585;765;780;960
s02,p02,hdt,sessions/s02.csv,0;180;212;392;405;585;601;781;795;975
```

`events` are the block boundaries in seconds on the session's own clock
(`start1;end1;start2;end2;…`), from the protocol events files. `file` is a CSV
(path relative to the manifest), with any sampling rate:

```csv
time_s,temple_bf,brain_hbo,scalp_hbo
0.0,0.12,0.03,-0.01
...
```

The NIRS columns should be the session-level, module-averaged ΔHbO traces the paper
correlates. Other layers or chromophores, such as `brain_hbt` or a short-SDS HbO trace,
can be added through `--layers`. `python simulate.py write OUT` writes an example
dataset in this format.

### Outputs (`--out`)

- `per_session.csv`: real r, mixed r and gap for each session, layer and analysis
- `summary.json`: everything in the printed tables
- `matrix_<protocol>_<analysis>_<layer>_<lag|zero>.csv`: the full session × session
  correlation matrix (rows: Temple-BF, columns: NIRS). The diagonal holds the real pairs.

## Files

| file | purpose |
|---|---|
| `shared_maneuver_test.py` | the analysis (numpy + scipy only) |
| `simulate.py` | synthetic sessions and the demo tables |
| `calibrate.py` | false-positive check on simulated null studies |
| `tests/` | unit and end-to-end tests |

## Sources

- Gulati D, Rudaeva A, Dutta A, Rogers D, Prajapat R, Kumar N, Gupta S, Goyal D,
  Boas DA. *Evaluating Brain Flow Index from a temple-worn wearable against
  depth-resolved time-domain NIRS during head-down tilt and postural transitions in
  healthy young men.* bioRxiv 2026. doi:10.64898/2026.09.17.751664. Methods Sec. 2.5–2.6,
  Table 1, Fig. 4, Limitations 2 and 6. The paper says the recordings are available
  from the corresponding author on reasonable request (research@temple.com).
- Nguyen T, Hoehl S, Vrtička P. *A Guide to Parent-Child fNIRS Hyperscanning Data
  Processing and Analysis.* Sensors 2021;21(12):4075 (random-pair analysis).
- Nastase SA, Gazzola V, Hasson U, Keysers C. *Measuring shared responses across
  subjects using intersubject correlation.* Soc Cogn Affect Neurosci 2019;14(6):667–685.
