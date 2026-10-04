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
| `p(gap)` | one-sided Wilcoxon signed-rank, gap > 0. **On the residual analysis, this is the test for session-specific tracking**; on full sessions it is descriptive (see Calibration) |
| `p(perm)` | permutation test: true pairing vs 10,000 random different-subject pairings (too liberal, reported for completeness; see Calibration) |
| `ident` | how many sessions' Temple-BF correlates best with *its own* NIRS trace, out of all sessions (chance ≈ 1) |
| brain − scalp | the paper's brain-vs-scalp contrast, repeated for real pairs, mixed pairs, and the gap |

**Residual analysis.** Before correlating, each signal has everything its own session's
block timing can explain regressed out. The regressors are the task-block
indicator passed through 1st- and 2nd-order lags with time constants of 2–80 s, plus a
trend. What is left is activity that is not locked to the maneuver. This is the closest
this dataset gets to Limitation 6, sensitivity to "smaller spontaneous changes typical
of daily life".

**Alignment.** Mixed pairs need a shared protocol clock. By default (`--align warp`)
each segment between event markers is stretched onto the median timing, so block
onsets line up across sessions. This gives mixed pairs the best chance of matching,
which is the conservative choice. `--align onset` only aligns the first event.

**Repeat participants.** Pairs from the same `subject_id` are never used as mixed pairs.

## A simulation grounded in the paper

Before anyone runs the test on real recordings, a simulation can say what to expect. To
be useful it has to be tied to the study, so every number in it comes from the paper,
from published physiology, or from a fit to the paper's published results. Nothing is
tuned by eye.

### Where every number comes from

| parameter | value | source |
|---|---|---|
| block design | 5 blocks; 180 s (tilt, supine), 90 s (squat); task in blocks 2 and 4 | paper, Sec. 2.3, Fig. 2 |
| brain response speed | first-order, τ = 14.4–28.9 s, drawn per session | paper, Sec. 3.1: half of the block peak in 10–20 s (τ = t½ / ln 2), peak at 60–90 s |
| spontaneous fluctuations | band-limited noise, 0.01–0.12 Hz | literature: very-low-frequency (~0.04 Hz) and Mayer-wave (~0.1 Hz) cerebral oscillations, Obrig et al. 2000 |
| analysis pipeline | 1 Hz, α = 0.35 EMA, ±30 s lag search, windows ±90 / ±45 s | paper, Sec. 2.5–2.6 |
| Temple-BF delay | fitted (median optimal lag 13.5–15 s in every consistent fit) | fitted to the paper's 13–15 s median optimal lag, Sec. 3.2 |
| size of session-specific brain variation | fitted | Table 1 and Table 2 brain-layer medians |
| Temple-BF's own noise | fitted | same |
| shared shape difference between Temple-BF and brain (a drift) | fitted | Fig. 4 group-mean r, which is below 1 |
| scalp response size and speed | fitted | Table 1 scalp median, Fig. 4 scalp group-mean r |
| transition gaps between blocks | 8–20 s | **assumed** (not reported); varied in the sensitivity runs |
| trackable share of brain variation (physiology vs inversion noise) | 50% | **assumed**; varied 25–100% in the sensitivity runs |
| *head-down tilt only:* fast share of the brain response | fitted (lands at 0.99) | bounded by Fig. 3A, where brain ΔHbO jumps almost at once when the table tilts |
| *head-down tilt only:* brain decline within a block | fitted, capped at 25% (lands at 14–25%) | Fig. 3A: brain ΔHbO falls about 12% from its peak by the end of each block |
| *head-down tilt only:* Temple-BF transient at each table movement | fitted (lands at about 1× the step) | proposed mechanism, see [Head-down tilt](#head-down-tilt-what-it-took-to-reproduce-it); Temple-BF uses heart-rate features (Sec. 2.2.1), and heart rate changes abruptly at each tilt (Fig. 4A) |
| **tracking share w** | 0, 0.5, 1 | **the unknown**: 0 = Temple-BF only follows the maneuver; 1 = it follows the session's own brain physiology |

### How the fit works (`fit_to_paper.py`)

1. For each protocol and each w, simulated sessions go through the paper's analysis.
   The free parameters are fitted so that six published statistics come out right:
   - median lag-adjusted r, brain and scalp layers (Table 1)
   - median zero-lag r, brain layer (Table 1)
   - median transition-window r (Table 2)
   - group-mean r, brain and scalp layers (Fig. 4)

   The fit also has to land the median lag at 13–15 s. The search uses differential
   evolution, a global, gradient-free method; a local search got stuck because the
   statistics are medians and lag maxima. The random draws are fixed across evaluations
   (common random numbers).
2. **Consistency with the paper is judged against the paper's own sampling error.** The
   standard error of each statistic is estimated from 30 simulated 20-session studies.
   A parameter set counts as consistent when its χ² distance to the paper (in
   standard-error units, 5 df) is at most 11.07, the 95% point.
3. Several parameter sets fit equally well, and they predict different gaps. So for each
   w the script searches the consistent sets for the **smallest and largest gap**: an
   identification set, meaning every gap between the two ends fits what the paper
   reports. At both ends it runs the full mixed-pair test on 40 simulated studies to get
   power.

### Two things the paper's numbers imply about its methods

- **The transition-window correlations were almost certainly lag-aligned.** For squat
  the paper reports transition r = 0.861, which is above its zero-lag full-session
  r = 0.701. With the 13–15 s lag, a zero-lag ±45 s window cannot exceed about 0.88 even
  with no noise at all. That leaves no room for the noise that pulls the full-session
  zero-lag r down to 0.701. The paper does align on the session's optimal lag for its
  partial correlations, so the simulation does the same for the transition windows.
- **The average Temple-BF and brain shapes differ.** Group-mean r is 0.915–0.941, not 1,
  and per-session r is about as high. If Temple-BF had exactly the brain's average shape,
  group-mean r would be close to 1, so the simulation includes a shared shape difference.

## What the paper's published numbers can and cannot tell us

Best fit per protocol and tracking share w (simulated statistics; paper in the first
row of each block). χ² ≤ 11.07 means consistent with the paper.

| protocol | w | lag-adj. r brain | lag-adj. r scalp | zero-lag r brain | transition r | group r brain | group r scalp | median lag (s) | χ² |
|---|---|---|---|---|---|---|---|---|---|
| squat | *paper* | *0.843* | *0.700* | *0.701* | *0.861* | *0.916* | *0.783* | *13–15* | |
| squat | 0 | 0.835 | 0.700 | 0.719 | 0.847 | 0.916 | 0.783 | 17.5 | **20.9** |
| squat | 0.5 | 0.837 | 0.699 | 0.704 | 0.862 | 0.917 | 0.783 | 14.0 | 1.7 |
| squat | 1 | 0.843 | 0.676 | 0.692 | 0.856 | 0.916 | 0.792 | 14.4 | 1.7 |
| supine | *paper* | *0.839* | *0.767* | *0.796* | *0.876* | *0.941* | *0.929* | *13–15* | |
| supine | 0 | 0.842 | 0.767 | 0.805 | 0.878 | 0.940 | 0.929 | 14.6 | 5.3 |
| supine | 0.5 | 0.840 | 0.767 | 0.800 | 0.867 | 0.941 | 0.929 | 14.0 | 4.1 |
| supine | 1 | 0.845 | 0.767 | 0.798 | 0.858 | 0.940 | 0.930 | 14.9 | 11.9 |
| head-down tilt | *paper* | *0.912* | *0.494* | *0.826* | *0.878* | *0.915* | *0.430* | *13–15* | |
| head-down tilt | 0 | 0.911 | 0.475 | 0.826 | 0.877 | 0.915 | 0.607 | 15.0 | 1.1 |
| head-down tilt | 0.5 | 0.909 | 0.483 | 0.829 | 0.879 | 0.915 | 0.578 | 14.0 | 1.3 |
| head-down tilt | 1 | 0.909 | 0.480 | 0.827 | 0.878 | 0.916 | 0.589 | 14.2 | 1.2 |

Head-down tilt rows use the extended model described [below](#head-down-tilt-what-it-took-to-reproduce-it);
the original model gave χ² = 407–579 there. The scalp group-mean r for head-down tilt
stays too high (0.58–0.61 against 0.430), because the scalp model is a single smooth
response and the paper's scalp trace drifts (Fig. 3A). The scalp does not enter the gap.

What this says:

1. **For stand-to-supine, the published statistics fit a device that only follows the
   maneuver as well as one that tracks the person.** No analysis in the paper can tell
   these apart. That is the gap Limitation 2 describes, and it is what the mixed-pair
   test measures.
2. **For stand-to-squat, a maneuver-only device fits everything except the lag.** To
   reproduce the drop from lag-adjusted r (0.843) to zero-lag r (0.701) it needs a 17–20 s
   delay, while the paper reports 13–15 s. A tracking device produces that drop at a 14 s
   lag, because the faster fluctuations it shares with the brain decorrelate when
   misaligned. This hints that Temple-BF carries session-specific information during
   squats, but the hint depends on the model. The robustness runs below test it, and only
   the mixed-pair test on real data can settle it.
3. **For head-down tilt, the published statistics also fit every tracking share equally
   well** (χ² 1.1–1.3), once the model includes the mechanisms in the next section. They
   also allow almost no session-to-session brain variation, so even a perfect tracker may
   have little to track there.
4. **The paper's brain-versus-scalp contrast does not need any tracking.** With w = 0, the
   brain-layer r beats the scalp-layer r (one-sided Wilcoxon, p < 0.05) in 100% of
   simulated studies for every protocol. A device that only follows the maneuver
   reproduces the paper's main contrast, because the brain layer's *average* response
   resembles the device's average response more than the scalp layer's does. Repeating
   the contrast on the gap (`brain − scalp` in the output) removes this.

## Head-down tilt: what it took to reproduce it

The original model failed for head-down tilt (χ² 407–579). The pattern it could not
produce is this:

| head-down tilt | paper | original model |
|---|---|---|
| per-session lag-adjusted r | 0.912 | 0.88–0.90 |
| transition-window r | **0.878**, below the full session | 0.92–0.95, above it |
| group-mean r | 0.915, about equal to per-session r | 0.92 |
| median lag | 13–15 s | 15–20 s |

In the paper, the ±90 s window around each tilt matches *worse* than the session as a
whole. So the mismatch between Temple-BF and the brain layer is concentrated at the
transitions, and it is shared by every session (group-mean r is no higher than
per-session r). The lag search absorbs any difference in response speed, so the mismatch
has to be a difference in *shape* at the transitions.

**Screening candidate mechanisms** (`results/hdt_screen.py`; each one globally fitted,
χ² in the paper's standard errors, threshold 11.07):

| candidate | χ², w = 0 | χ², w = 1 |
|---|---|---|
| Temple-BF smooths more slowly + part of the brain response is fast | 492 | 474 |
| brain response adapts (falls back) within each block | 404 | 540 |
| **Temple-BF has a transient at each table movement** | 43 | 66 |
| transient + unconstrained adaptation | 30 | 10.4 |
| **transient + brain shape bounded by Fig. 3A** (fast rise, at most 25% decline) | **1.7** | **1.5** |

Only candidates with a Temple-BF transient come close. Unconstrained, the fit also
wanted the brain response to fall back by 75% within each block. Fig. 3A shows about
12%, so that solution was rejected and the decline was capped at 25%. With the brain
shape bounded by the figure, the full fit reproduces all four brain-layer statistics
and the lag at every tracking share (χ² 1.1–1.3).

**The proposal.** During head-down tilt, Temple-BF probably shows a short excursion at
each table movement, about the size of the block response itself, that the brain layer
does not show. A plausible source is in the paper: Temple-BF includes heart-rate and
pulse-waveform features (Sec. 2.2.1), and Fig. 4A shows heart rate changing abruptly at
every tilt. Temple can check this directly with data they already have:

1. Average Temple-BF (and heart rate) in a ±30 s window around each table movement,
   aligned on the events file or the accelerometer, and compare with brain-layer ΔHbO.
   The model predicts an excursion in Temple-BF that peaks about 10–25 s after the
   table starts moving and is absent from the brain layer.
2. Re-run the paper's correlations with those windows masked. If the transient is the
   cause, the transition-window r should rise towards the full-session r.

This is the weakest part of the simulation. The extended head-down tilt model has 8
fitted parameters against 5 published statistics, so a good fit shows the mechanism
*can* produce the paper's numbers, not that it does. The check above is the way to find
out.

## What the test would show on real recordings

Range of outcomes consistent with the paper, brain layer, lag-adjusted, 20 sessions.
Power is the share of 40 simulated studies with p(gap) < 0.05. The residual test is the
calibrated one (see Calibration).

| protocol | w | full-session gap | own trace matched best (of 20) | residual gap | residual-test power |
|---|---|---|---|---|---|
| squat | 0 | 0.001 | 1 | −0.003 | 0.05 (false positives) |
| squat | 0.5 | 0.046 – 0.076 | 11 – 18 | 0.17 – 0.43 | 1.00 |
| squat | 1 | 0.038 – 0.094 | 12 – 20 | 0.12 – 0.36 | 0.97 – 1.00 |
| supine | 0 | 0.001 | 1 | −0.008 | 0.03 (false positives) |
| supine | 0.5 | 0.001 – 0.008 | 2 – 5 | −0.008 – 0.009 | 0.00 – 0.15 |
| supine | 1 | 0.015 – 0.019 | 11 – 12 | 0.05 – 0.08 | 1.00 |
| head-down tilt | 0 | −0.001 | 1 | 0.000 | 0.00 (false positives) |
| head-down tilt | 0.5 | 0.000 – 0.010 | 1 – 6 | −0.002 – 0.028 | 0.00 – 0.85 |
| head-down tilt | 1 | 0.000 – 0.030 | 1 – 19 | −0.002 – 0.166 | 0.00 – 1.00 |

Head-down tilt is the least predictable routine. Its statistics allow the
session-specific brain variation to be anywhere from almost none (brain variation SD
0.012 of the step) to moderate (0.20). At the low end there is nothing for even a perfect
tracker to follow, and the test shows no gap.

Gaps look small in units of r because the headline r is already high: the maneuver
alone gives r ≈ 0.76–0.84 here. The identification count is easier to read. A device
that tracks the session should pick out its own session's brain trace in 11–20 of 20
squat sessions, against 1 expected by chance.

### Sensitivity to the assumptions (w = 1)

Each row changes one assumed value and redoes the whole fit.

| assumption changed | squat gap | squat residual power | supine gap | supine residual power | χ² squat / supine |
|---|---|---|---|---|---|
| none (main run) | 0.038 – 0.094 | 0.97 – 1.00 | 0.015 – 0.019 | 1.00 | 1.7 / 11.9 |
| trackable share 25% | 0.011 – 0.045 | 0.20 – 1.00 | 0.001 – 0.013 | 0.00 – 0.75 | 0.0 / 5.7 |
| trackable share 100% | 0.012 – 0.063 | 0.90 – 1.00 | 0.001 – 0.017 | 0.00 – 1.00 | 0.0 / 8.1 |
| fluctuations down to 0.003 Hz | 0.009 – 0.073 | 0.28 – 1.00 | 0.010 – 0.035 | 0.70 – 1.00 | 0.0 / 3.7 |
| transition gaps 5–30 s | 0.018 – 0.062 | 0.90 – 1.00 | 0.003 – 0.013 | 0.07 – 0.88 | 1.2 / 2.2 |

The squat result for a maneuver-only device (w = 0) also holds when the assumptions
change. It stays inconsistent with the paper because it needs a 17–18 s lag: χ² = 20.5
with fluctuations down to 0.003 Hz, and 15.1 with transition gaps of 5–30 s. For supine,
w = 0 stays consistent (χ² 3.7–5.1).

What to expect on the real data, then:
- If Temple-BF tracks the session during squats, the test is likely to show it. The
  residual test detects it in most simulated studies under most assumptions; the
  low-end exceptions are when little of the brain variation is trackable.
- For stand-to-supine the expected gap is smaller, and detection depends on the
  assumptions.
- Head-down tilt could show anything from no gap to a clear one. A null result there
  would say little about the device.
- **Squat is the protocol to run first**, then supine, then head-down tilt.

### Calibration

How often does the test report a gap when the device only follows the maneuver? This
uses the w = 0 fit, over 100 simulated null studies (`python calibrate.py --studies 100`),
and shows the share with p < 0.05:

| protocol | analysis | transition gaps 8–20 s: p(gap) | p(perm) | gaps 5–30 s: p(gap) | p(perm) |
|---|---|---|---|---|---|
| squat | full, lag-adjusted | 0.06 | 0.22 | **0.34** | **0.67** |
| squat | residual, lag-adjusted | 0.04 | 0.08 | 0.02 | 0.05 |
| supine | full, lag-adjusted | 0.03 | 0.12 | 0.09 | **0.25** |
| supine | residual, lag-adjusted | 0.01 | 0.03 | 0.09 | 0.09 |
| head-down tilt | full, lag-adjusted | 0.00 | **0.44** | 0.00 | **0.77** |
| head-down tilt | residual, lag-adjusted | 0.00 | 0.06 | 0.00 | 0.24 |

With 100 studies, each rate has a Monte Carlo error of about ±2–3 points.

(`python calibrate.py --studies 100 --gap 5,30` for the right-hand columns; full output,
including zero-lag rows, in `results/calibration_*.txt`.)

**This changed the method.** On full sessions, a real pair shares its own exact
transition timing, and mixed pairs cannot fully reproduce that even after time-warping.
When transition durations vary a lot, that timing alone produces a small but
"significant" gap for a device that only follows the maneuver: up to 34% false positives
for squat. The residual analysis regresses each session's own timing out, and its
p(gap) stays near the nominal 5% (0–9%). So:

- **The residual p(gap) is the test for session-specific tracking.**
- The full-session real / mixed / gap numbers describe how much of the headline the
  maneuver explains, but they are not used for inference.
- The permutation p value is too liberal and is reported only for completeness.
- On real data, the events files give the actual transition durations. Rerunning
  `calibrate.py --gap` with their observed range shows how much the full-session test
  can be trusted.

## How to read a result on the real recordings

- **A clear residual gap on the brain layer** means Temple-BF carries information about
  the individual session beyond the imposed maneuver and its timing. That strengthens the paper's claim.
  A larger gap for brain than for scalp would be a stronger version of the paper's
  brain-vs-scalp argument.
- **A small gap does not mean the device fails.** It can mean that people's brain
  responses to a tilt are very similar, so there is little session-specific signal to
  track (`python simulate.py demo`, row `person*`).
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
- **The simulation matches the paper's medians, not its spread.** The violin plots in
  Fig. 4 show more session-to-session variation than the simulation has. For example,
  the paper's stand-to-supine brain-minus-scalp difference was not significant
  (p = 0.082), but it is significant in every simulated study. So the power figures
  above are best cases.
- **Head-down tilt needs extra, proposed mechanisms** (see above). Its predictions rest
  on a model with more free parameters than published statistics.
- **The simulation is a model.** The response shapes are first-order, the device is a
  linear mix, and the fluctuations are band-limited noise. The fits show this model is
  consistent with the paper's published numbers for squat and supine, not that it is
  correct.

## Usage

```bash
pip install -r requirements.txt

python -m pytest                              # tests
python fit_to_paper.py --protocols squat,supine   # fit to the paper, gap ranges, power (~25 min on 4 cores)
python fit_to_paper.py --protocols hdt --search-effort 2 --out results/fit_hdt.json   # ~40 min
python results/merge_fit.py results/fit_hdt.json hdt
python fit_to_paper.py --mixes 1 --phys-share 0.25 --out results/sens_q025.json   # a sensitivity run
python calibrate.py --studies 100 [--gap 5,30]   # false-positive check (~3 min)
python simulate.py demo                       # one simulated study per device, fitted parameters

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
| `simulate.py` | synthetic sessions; every parameter labelled with its source |
| `fit_to_paper.py` | fits the simulation to the paper's published statistics; gap ranges and power |
| `paper_fit.json` | the main fit (used by `simulate.py demo` and `calibrate.py`) |
| `results/` | sensitivity runs, calibration output, the head-down tilt screen (`hdt_screen.py`) and the fit used for head-down tilt (`fit_hdt.json`, merged into `paper_fit.json` by `merge_fit.py`) |
| `calibrate.py` | false-positive check on simulated null studies |
| `tests/` | unit and end-to-end tests |

## Sources

- Gulati D, Rudaeva A, Dutta A, Rogers D, Prajapat R, Kumar N, Gupta S, Goyal D,
  Boas DA. *Evaluating Brain Flow Index from a temple-worn wearable against
  depth-resolved time-domain NIRS during head-down tilt and postural transitions in
  healthy young men.* bioRxiv 2026. doi:10.64898/2026.09.17.751664. Sec. 2.3–2.6, 3.1–3.2,
  Tables 1–2, Figs. 2–4, Limitations 2 and 6. The paper says the recordings are available
  from the corresponding author on reasonable request (research@temple.com).
- Obrig H, Neufang M, Wenzel R, Kohl M, Steinbrink J, Einhäupl K, Villringer A.
  *Spontaneous low frequency oscillations of cerebral hemodynamics and metabolism in
  human adults.* NeuroImage 2000;12(6):623–639. doi:10.1006/nimg.2000.0657
  (frequency bands of spontaneous fluctuations).
- Nguyen T, Hoehl S, Vrtička P. *A Guide to Parent-Child fNIRS Hyperscanning Data
  Processing and Analysis.* Sensors 2021;21(12):4075 (random-pair analysis).
- Nastase SA, Gazzola V, Hasson U, Keysers C. *Measuring shared responses across
  subjects using intersubject correlation.* Soc Cogn Affect Neurosci 2019;14(6):667–685.
