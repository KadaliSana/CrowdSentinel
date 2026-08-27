# Crowd-Crush Precursor Detection: Literature Gaps and a Proposed Model

Status: **design proposal for review — not implemented.**
Companion: `REFERENCES.md` (92 refs, verification-tagged).

## 1. How this review was done

Searches were run across crowd physics, crowd anomaly detection, video anomaly
detection, crowd counting, face detection, tracking, TinyML/edge, and post-incident
disaster analyses. Papers marked `[F]` in `REFERENCES.md` were downloaded and read;
`[M]` are metadata/abstract-level; `[G]` are paywalled and still needed.

**Honesty note.** Claims below drawn from `[M]`/`[G]` entries are abstract-level. The
five `[G]` papers (esp. D1, D2, C1, G3, D6) must be read before submission — D1/D2 are
the field's own gap statements, and C1/D6 are the closest prior art to what is proposed.

## 2. What the literature establishes

**2.1 The physics side has a validated predictor, and it is not density.**
Johansson, Helbing, Al-Abideen & Al-Bosta [A1], analysing video of the 2006 Jamarat
disaster, tested density, speed, flow, divergence and curl and found **none** of them
identified the accident. The quantity that did was *crowd pressure*:

    P = rho * Var(v)        (density times the variance of speeds)

Turbulent crowd motion began when P exceeded **0.02 s^-2**, roughly **10 minutes before**
the crush; average flow had dropped below **0.8 ped/m/s** more than **30 minutes** before
[A1]. The authors explicitly propose that these "advance warning signs ... could be
evaluated on-line by a video analysis" — a suggestion largely unrealised in the CV
literature 15+ years later.

Post-incident work corroborates the density regime: Itaewon 2022 reached ~7.57 ped/m^2
average and 9.95 peak, with crowd pressure ~1063 N/m [B1].

**2.2 The computer-vision side builds classifiers, not physical measurements.**
Modern stampede detectors [C1, C2, C3] and crowd anomaly systems [D3, D4, D5] learn a
*label* ("stampede"/"normal", or an anomaly score) from optical flow or detections. The
canonical baseline remains the Social Force Model [A3]. These systems are deployable but
produce a scene- and dataset-specific score with no physical units, so their thresholds
cannot be transferred between cameras, nor compared with the safety literature.

**2.3 Detection-based sensing fails precisely in the danger regime.**
Occlusion accounts for the majority of missed detections in crowds [H7, F11], with
reported degradation of 30-40% even for state-of-the-art methods under heavy occlusion,
and 40-60% missed individuals when tightly packed. Face-based sensing is worse still,
since heads turn away as density rises [G3]. Density-map regressors degrade more
gracefully but still degrade [F11].

**2.4 Edge deployment exists for counting, not for risk.**
MiCrowd [I1] demonstrates crowd *counting* on an MCU; TinyML surveys [I2, I3] establish
the constraint envelope (<512KB SRAM, <1 GOP/s, <1W). Edge video-anomaly reviews [E4]
focus on offloading. No crowd-*crush precursor* system is demonstrated at MCU-NPU scale.

## 2.5 Prior-art check (VERIFIED against full texts, 2026-08-27)

The novelty claims were tested directly against the five gated papers, now read in full.

| Paper | Mentions crowd pressure / Helbing-Johansson predictor? | Evidence |
|---|---|---|
| **C1** Stampede detector w/ dense optical flow, *Eng. Appl. AI* 142 (2025) | **NO** — zero occurrences of "crowd pressure", "Helbing", "Johansson", "variance of speed", "calibration", "homography" | keyword scan of full text |
| **C2** Stampede detection CNN-LSTM + Farneback, *Sci. Reports* (2026) | **NO** — cites Helbing only for the 1995 *social force* escape-panic model, not pressure | 2 hits, both social-force |
| **D2** *Crowd anomaly estimation and detection: A review*, Franklin Open 8 (2024) | **NO** — zero occurrences in an entire review of the field | keyword scan of full text |
| **D6** div-curl flow fields, *Pattern Recognition* 88 (2019) | **NO** — and it *misreports* the source: states Johansson identifies hazards "by comparing the density, divergence, and curl" | line 139 |
| **D1** *Deep crowd anomaly detection*, Sharif, Jiao & Omlin, *AI Review* 58:139 (2025) | — | survey; see S3 |

**This is the strongest single result of the review.** D6's misstatement is the crux: Johansson
et al. [A1] tested density, divergence and curl and found that **all three failed** to indicate the
Jamarat disaster; pressure was the quantity that worked. The closest prior art therefore cites the
foundational study while inverting its central finding, and the 2024 review of the field never
mentions the predictor at all. Gap G1 is not inferred — it is demonstrated.

**Contribution 1 is not preempted** by any of the four candidate competitors.
**Contribution 2: not preempted, on substitute evidence.** G3 (the only direct threat) proved
unobtainable in full text, so the claim is deliberately *not* rested on it. Two accessible papers
carry the argument instead, and they make it stronger:

- **G10 (DAFE-FD, CVPR 2019)** estimates a crowd **density map in order to improve face
  detection** — i.e. density -> detectability. Contribution 2 runs the same coupling **backwards**:
  detector dropout -> density. The relationship is established in the literature; the inverse use
  of it is not.
- **G11 (Repulsion Loss, CVPR 2018)** shows experimentally that "the detector is harmed by crowd
  occlusion", giving quantitative grounding for the occlusion index.

G3's abstract indicates it *engineers occlusion away* (training a detector robust to 90 deg rotation
and 25/50/75% occlusion). That is the field's standard response and precisely the assumption
Contribution 2 inverts: the literature repairs the sensor, SF-CPI instruments its failure. Cited at
abstract level and flagged `[A]`; no claim depends on it.

## 3. Gaps

**G1 — The physics/CV disconnect.** The one predictor validated against a real crowd
disaster [A1] is essentially absent from the deep-learning stampede literature
[C1-C3, D3-D5], which optimises dataset labels instead. Nobody computes P = rho*Var(v)
online and compares it against the published 0.02 s^-2 threshold.

**G2 — Calibration is assumed to be the blocker.** Computing rho in ped/m^2 is taken to
require ground-plane homography, which is impractical for ad-hoc cameras. This
assumption pushes practitioners toward unitless learned scores (G1). *We show in S4.1
that it is false for pressure specifically.*

**G3 — Evaluation rests on acted panic.** Confirmed in the field's own words: D1 states the
Minnesota dataset "is straightforward where the performance of methods is saturated on it",
and that Ped1/Ped2 are single-location, fixed-camera and very low resolution. C1's own new
stampede datasets contain **up to 6 people (GSMADC) and a minimum of 15 (GBA)** while
reporting ~99% on UMN/PETS-2009 — i.e. the state of the art is validated on small acted
groups, not crowds at crush density. Real crush events are analysed only post-hoc [B1-B4].
*Mitigation available:* C1 released GBA-Stampedes and GSMADC publicly; both should be
adopted as additional baselines despite their scale limitation.

**G4 — Systems do not report sensing reliability.** A detector-based risk score that
silently degrades with density is safety-inverted: the alarm weakens as danger grows.
No reviewed system outputs a confidence that is a function of its own occlusion regime.

**G5 — No MCU-class demonstration of risk (not counting) inference.** D1 lists this explicitly
as two of its open challenges — "lack of hardware applications" ("crowd anomaly detection
demands the processing of a large amount of video data, which needs a powerful GPU") and
"lack of computing power".

**G6 — Anomaly is defined subjectively, per dataset.** D1's first open challenge: "Anomaly
definition is fully subjective. Based on the time and place, the same event can be either normal
or abnormal", with popular datasets treating anything unseen in training as anomalous. A
physically-defined criterion in s^-2, comparable across scenes and against published thresholds,
is a direct answer to this — and is what S4.1 makes possible without calibration.

## 4. Proposed model

Working name: **SF-CPI — Scale-Free Crowd Pressure Index**, with occlusion-aware density
and confidence gating.

### 4.1 Contribution 1: crowd pressure is scale-invariant (addresses G1, G2)

Dimensional analysis of Helbing's quantity:

    [rho] = L^-2 ,  [Var(v)] = L^2 T^-2  =>  [P] = T^-2

P carries **no length dimension** — which is why the published threshold is quoted in
s^-2 [A1]. Consequently, for an image cell with unknown metres-per-pixel scale s:

    rho_m   = N / (s^2 * A_px)
    Var_m(v)= (s*fps)^2 * Var_px(v)
    P_m     = rho_m * Var_m(v) = (N / A_px) * fps^2 * Var_px(v) = P_px

**The calibration cancels exactly.** Crowd pressure can be computed from uncalibrated
video, in correct physical units, given only the frame rate — and compared directly with
the 0.02 s^-2 literature threshold. (Verified numerically: four scales spanning 50x give
identical P.) Because s cancels *per cell*, perspective variation across the image is
handled by computing P per cell rather than globally, which also localises risk.

This reframes the problem: the barrier to physically-grounded risk from uncalibrated
video is **not calibration** — it is **counting N**. That is a sharper and more tractable
problem statement than the literature currently uses.

### 4.2 Contribution 2: occlusion-aware density under detector dropout (addresses G4)

N per cell is estimated from two partially-independent sensors:
- **N_det** — face/person detections (accurate when sparse, collapses when dense)
- **M_flow** — moving-occupancy from dense optical flow (survives occlusion, cannot count)

Define an **occlusion index** from their disagreement, e.g. omega = 1 - N_det/f(M_flow),
calibrated per scene during a calm baseline window. Rising omega at constant or rising
M_flow indicates densification *and* that the detector is dropping out. omega both
(a) corrects N and (b) produces the confidence term below.

### 4.3 Contribution 3: confidence-gated, fail-loud risk output (addresses G4)

Risk is emitted as a level plus an explicit confidence derived from omega, with
hysteresis to prevent flapping. Crucially, when sensing confidence collapses the system
**escalates uncertainty rather than lowering risk** — the opposite of the silent failure
in S2.3. This is a safety-engineering property absent from the reviewed systems.

### 4.4 Contribution 4: MCU-class demonstration (addresses G5)

SCRFD [G1] runs on the RTL8735B NPU at 576x320 producing boxes and 5 landmarks; the host
computes flow and pressure. Reports feasibility, latency and power at a cost/power point
far below the Jetson-and-above assumption of [C1-C3].

### 4.5 Demoted: head-orientation divergence

Originally proposed as the headline contribution. **Partially preempted** — US Patent
11,509,831 [K1] already claims synchronous head-direction alerting, and motion/head
mismatch appears in prior work [K5]. Retained only as an exploratory auxiliary signal,
to be reported honestly as such, and only if S5 shows it adds measurable value.

## 5. Evaluation protocol

Baselines: Social Force Model [A3], flow-variance, and a learned stampede detector
[C1/C3] where reproducible. Ablations over {P alone, P+omega, P+omega+head}.
Metrics: **detection latency before the event** (the operationally meaningful number,
following [A1]'s 10-minute lead time), AUC/AP, and false-alarm rate per hour.
Data: UMN + UCSD for comparability (acknowledging G3), plus dense real crowd footage;
report the acted-vs-real limitation explicitly rather than claiming transfer.
A file-replay source path is required so all of this runs without the board.

## 6. Honest limitations

- P is validated at one real disaster [A1]; the 0.02 s^-2 threshold is R-dependent and
  should be treated as an order-of-magnitude reference, not a constant.
- The scale-cancellation assumes motion is approximately in the ground plane and that s
  is locally constant within a cell; strong tilt/foreshortening violates this.
- 512 kbps 720p compression injects artifact motion into flow — a measurable accuracy tax.
- Counting N remains the dominant error source; SF-CPI reframes it, it does not solve it.
- No real crush video benchmark exists; conclusions about crush precursors are inferential.
