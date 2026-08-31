# A Portrait of the Language Model as an AM Receiver

> **A human disclaimer.** The AI has produced stunning achievements in applied
> mathematics. I found myself between jobs for a weekend, so I decided to join the AI
> research train before I became busy again. I directed Claude Fable to work on an old idea of mine
> from the pre-transformer days, whether DNNs are some form of communication channel,
> in the EE sense. The answer appears to be a conditional yes. Too bad too many years have passed
> and I can no longer fully judge its work, but here it is. — ivanamies

**TL;DR.** Per-sender amplitude carries the message (+22% loss to strip it). Found inside: AGC, a learned
pilot tone that readers demodulate, squelch, a free limiter, code-locked tuning, a half-fixed
mixer. A tiny LM grows with the whole radio installed for 0.72 nats.

*(257 characters.)*

## Notation, units, and the cast

- **The stream / wire / channel** — the residual stream, one object. **Tap N** = the stream read
  after block N (GPT-2-small: 12 blocks; taps 3/6/9). **Writers** (senders): the attention heads
  and MLPs adding into it; **readers** consume it through a LayerNorm.
- **AGC** — automatic gain control: LayerNorm's per-token division by the stream's own scale.
  **The pilot** — a small set of dimensions carrying the stream's scale *through* LayerNorm: any
  direction with stable pre-LN amplitude c re-emerges post-LN at c/σ(t). **TPC** — transmit power
  control: standardizing each writer's per-token output norm. **LO** — local oscillator: a
  *content-independent* fixed map; here, the fixed-linear part of inter-layer transport.
- **Costs** are in nats of validation cross-entropy on held-out text; a model "pays X nats" means
  its loss is X above the matched baseline. Injections respect the linearity budget of [1].
- **Seeds:** frozen-model measurements — independent probe redraws; from-scratch training (§9–8) —
  training seeds, matched token budgets. Three seeds throughout.
- **Outlier / massive dims:** dimensions with far-above-median mean magnitude, the known extreme
  activations of [5]. **A** — the amplification LayerNorm applies to ablation survivors, measured
  as the ratio of per-token normalization denominators between a clean and an ablated forward.

## 0. Background and prior work

This program's channel paper [1] measured the residual stream as a multiple-access channel; along
the way it found that roughly half of ablation "self-repair" [3, 4] is LayerNorm [2] acting as
gain control — a result restated in full in §3, because this paper is written to stand alone.
The question here is the literal follow-up: if the receiver has AGC, where is the rest of the
radio? Channel-census rows reused in §2's table (fading, linearity, overlap, the estimation bound,
despreading) carry one-line methods there and full restatements in Appendix C. Amplitude-coded information at the per-writer level connects to superposition
[6]; the pilot dimensions this paper identifies are the massive activations / outlier dimensions
known to quantization and serving practice [5], here given a functional account; attention sinks
[5c] are the parked receivers of the tuning census. The engineering alternatives to trained
normalization — removing it with careful scaling — are the normalizer-free line [7]; this paper's
training-time experiments instead *add* designed radio components to a standard recipe and measure
what the optimizer does with them. Models and corpora: GPT-2 [8a], Pythia [8b], TinyStories [8c].

## 1. What this paper establishes

| claim | where |
|---|---|
| Amplitude variation is signal, at the per-writer level: the transmitter has no power knob | §4 |
| LayerNorm does not destroy the scale it divides out — it re-encodes it, and training concentrates it onto a learned pilot | §5 |
| Readers demodulate against that pilot: two-thirds of its functional value is amplitude demodulation | §6 |
| The full dial panel, graded: a limiter is free, squelch is real but interference-limited, tuning is parked-or-code-locked, and the mixer is half a fixed translator — fully fixed at initialization | §7, §8 |
| The fixed translator can be kept through training at a measured price: 1.27 nats alone, 0.72 nats with companion wires — the wires are strongly sub-additive | §8, §9 |
| A tiny LM grows with every dial installed by construction, for 0.72 nats; passive wires are not adopted; stressing the stream causally grows outlier ballast | §9, §10 |

## 2. The portrait, in one table

Each row: the AM-system feature, the measurement, the number, the backing JSON (all JSONs are in
this repository). Rows marked [1] originate in the program's channel census and are restated in
Appendix C so this paper stands alone.

| AM-system feature | measurement | number | JSON |
|---|---|---|---|
| The message rides the amplitude envelope | strip write-norm variation → task collapses | NLL +22.3%; heads-only +11.0% vs MLPs-only +3.9% | tpc_retrofit |
| Transmitter power varies *with* the message | trained writers hold amplitude open | write-norm CV 0.38–0.66 vs 0.10–0.16 at init | tpc_census |
| Receiver AGC rides the level [1] | LN masks ablation damage; nothing left once amplitude is pinned | masking 2–8×; 3.92 → 0.93 under TPC | macl_l4_agc, tpc_retrofit |
| Carrier/pilot gives absolute scale | post-LN probe recovers pre-LN σ; training concentrates it | R² 0.91–0.99; top-16 dims 0.79–0.83 vs 0.04–0.19 random | pilot_exists |
| **The variation is read: demodulation** | shuffle the pilot dims; the TPC interaction rules out sink plumbing | +194% NLL; interaction 0.341 ± 0.004 | pilot_demod |
| The band is licensed and full | injected reference tone absorbed, not free | gain 0.05–0.07 by mid-depth; NLL +1.3–1.4% | pilot_retrofit |
| Fading channel [1] | natural-context path gains | 33–41% of contexts ≥3 dB down, 12–23% ≥10 dB | macl_l4pre_fading_natural |
| Transmitter linearity / IM spec [1] | two-tone intermodulation budget | IM <0.1 to 3× (taps 3/6), 10× (tap 9); 0.41 at 100× | wire_datasheet |
| Crowded band, many stations [1] | deliberate overlap above null at 32:1 overload | whitened overlap 1.3–33×; serial 3.14–4.12 vs parallel 0.64–1.59 | macl_l1_whitened, generalize_e3 |
| Co-phased transmitters [1] | ablation drops σ harder than power-additive | 2.6–3.4× the independence prediction | agc_law |
| Receiver at the noise limit [1] | whitened matched filter vs Cramér–Rao | 93–100% of the bound | crb_receiver |
| Dynamic range / peak limiting [1] | heavy-tailed level spikes | 34% of tokens beyond the χ² 99th percentile (init: 1–4%) | wire_datasheet |
| Spread + despread [1] | per-feature SNR vs recovery | −28 to −30 dB → AUROC 0.993–0.995 | macl_l15_despreading |
| **The radio can be grown** | a tiny LM trained with every dial installed | 0.72 nats; wires sub-additive | bonsai_score |

Two places the analogy is loose, stated so nobody over-reads it. **No carrier wave**: nothing
oscillates; "AM" means amplitude-coded, and since symbol identity rides *direction* while intensity
rides amplitude, the modulation is polar/QAM-shaped, not pure AM. **Broadcast vs multiple access**:
AM radio is one station per frequency; this is 156 simultaneous senders on one wire.

The defensible sentence, every clause carrying a number, a control, and a JSON: **the residual
stream is an amplitude-modulated multiple-access channel with receiver-side AGC, a learned pilot,
and readers that demodulate against it — and a small model can be grown with the whole radio
installed.** The chain that got here: Hydra confound → AGC measured → the gain law →
amplitude-is-signal → SGD opens the amplitude channel → the pilot exists → the pilot is used →
the radio is grown.

**Plainly.** This table is the paper. Each row is one thing a radio engineer would demand of an
"it's a radio" claim, the experiment that checked it, and where the raw numbers live.

## 3. The receiver has AGC, and half of "self-repair" is it

**Claim.** Restated here in full so this paper stands alone (first reported in this program's
channel paper [1]; same JSONs). LayerNorm is automatic gain control: ablate a branch, the stream's
per-token scale drops, every downstream LayerNorm divides by the smaller denominator, and the
survivors are amplified. Measured by re-running ablations with every denominator frozen to its
unablated per-token value: non-dominant branch damage grows 2–8× (per-seed medians 2.0/2.1/6.1) —
roughly half the celebrated ablation "self-repair" is receiver-side gain compensation, not
redundancy. The masking follows a gain law only loosely: measured amplification exceeds the
independence prediction (1−f)^(−1/2) by a consistent 2.6–3.4× because senders anti-cancel, and in
serial-residual models the masking ratio tracks measured amplification (Spearman 0.79) while
parallel models pin near 1.3 and decouple (−0.04).

**Method.**
- Frozen-denominator re-ablation on mechanically-located induction heads:
  `ecc_repr/agc.py::agc_census`, hook `agc.py::LNFreeze`.
- The gain law — per-LN σ ratios paired with an independent writer-decomposition estimate of each
  branch's variance share: `ecc_repr/agc_law.py::agc_law`.

**Controls.** *Positive:* a synthetic unit test — deleting a 50%-variance independent component
must amplify survivors by √2, and does; the all-branches ablation verifies full-damage detection.
*Negative:* a frozen *clean* forward reproduces the clean score to 3×10⁻⁷; LayerNorms upstream of
the write show amplification exactly 1; and the random-init rerun shows the damage-surface *fit*
is unrelated to the gain mechanism (the sum-shaped fit persists frozen, 0.996 vs 0.046). *Loading:*
mean-ablation, not zero-ablation; gain and damage measured on identical batches, per-token paired.

**Result.** As claimed; the dominant branch is the exception (0.684 live vs 0.544 frozen). The
usable residue is a two-forward-pass diagnostic: run one clean and one ablated forward, read A
from the σ ratios, and know — with no damage evaluation — which single-ablation numbers understate
damage, with ~1.3× as the observed floor even where masking is flat. Raw data:
`artifacts/results/macl_l4_agc.json`, `agc_law.json`.

**Scale & generality.** GPT-2 (+ random-init control) for the decomposition, 3 seeds; the law on
GPT-2 + Pythia-70m/160m, 54 branch-rows. The gain law generalizes across wiring; the damage link
is architectural.

**Images.** None committed yet (honest gap).

**Plainly.** Cut a cable inside the network and the damage looks small — partly because an
automatic volume control turns the survivors up before anyone measures. Hold that knob still and
the same cut costs two to eight times more. This is the observation the whole radio investigation
grew from: if the receiver has gain control, what else from the radio is in there?

## 4. Amplitude is signal: the transmitter has no power knob

**Claim.** Standardizing every writer's per-token output norm at inference — mechanically
successful (per-LN σ coefficient of variation collapses 1.5647 → 0.1007) — costs **+22.3%**
validation NLL. Heads-only regulation costs +11.0% against MLPs-only +3.9%: head amplitudes are
the information-dense ones. And SGD *opens* this channel: trained writers' per-token norm CV is
0.38–0.66 against a random-init 0.10–0.16 — training moves away from constant power, on every
model and corpus tested. With writer amplitudes pinned, the frozen-LN damage-masking ratio of [1]
collapses from 3.92 to 0.93: the AGC exists to ride exactly this informational variation.

**Method.**
- Per-writer deviation-norm standardization with a β knob, heads and MLPs separately and jointly:
  `ecc_repr/tpc.py::TPC`, driver `tpc.py::a2`.
- The writer-norm census, trained vs random-init, two corpora: `tpc.py::a1` with
  `generalize.py::GenWriterTap`.

**Controls.** *Positive:* the regulation demonstrably lands (σ CV collapses 16×), so the cost is
not an inert manipulation's. *Negative:* random-init models for the census; β=0 is verified to be
the identity (`tests/test_level_plan.py`). *Loading:* identical evaluation tokens across
conditions; calibration and evaluation disjoint.

**Result.** As claimed. Raw data: `artifacts/results/tpc_retrofit.json`, `tpc_census.json`.

**Scale & generality.** Retrofit: GPT-2, 3 seeds. Census: GPT-2 + Pythia-70m/160m, trained vs
random-init, TinyStories and WikiText. Scope note, so this is not quoted against the companion
paper's "the wire is FM": that claim is about the token's *total* norm (which carries almost
nothing); this one is about *per-sender* envelopes (which carry a fifth of the task). The stream is
amplitude-modulated per sender and level-normalized in aggregate.

**Images.** None committed yet (honest gap); the numbers are in the JSONs above.

**Plainly.** We tried to give the transmitter a power regulator, like a radio station holding
constant wattage. The model got dramatically worse — because the "wattage" wobble was never
sloppiness, it was the broadcast. An AM transmitter has no power knob separate from the message;
neither does this system, and its own training actively widens the amplitude swing.

## 5. The pilot exists, and training builds it

**Claim.** The pre-LN per-token scale is recoverable from the post-LN stream at R² 0.91–0.99 on
every trained model tested — LayerNorm re-encodes what it divides out. Training's contribution is
*concentration*: about 16 dimensions carry most of the recovery (top-16 R² 0.79–0.83 at shallow
taps, against 0.04–0.19 for random-16), while a random-init model holds the same information
diffusely (top-16 ≈ random-16). The carriers are the familiar massive/outlier dimensions [5],
functioning as a pilot tone.

**Method.**
- Ridge probe on the actual reading LayerNorm's output predicting log σ of its input, held out;
  localization via top-|mean| dims vs random dims: `ecc_repr/pilot.py::b1`.

**Controls.** *Positive:* the mechanism is guaranteed for a constant direction (c/σ(t) algebra), so
the probe must fire — total R² validates the probe. *Negative:* shuffled-pairing null (≈0);
random-init model — which carries the information but *unlocalized*, isolating what training adds.
*Loading:* same tokens; held-out splits; per-model reading LNs at matched depth fractions.

**Result.** As claimed; on parallel-residual models the pilot delocalizes with depth (top-16 falls
to 0.09–0.23 deep) while total recovery stays ≥0.98. A retrofit attempt — injecting an artificial
pilot on a spare direction at inference — fails on every registered criterion (Appendix A): the
niche is occupied. Raw data: `artifacts/results/pilot_exists.json`, `pilot_retrofit.json`.

**Scale & generality.** GPT-2 (+ random-init control), Pythia-70m/160m; 3 seeds.

**Images.** None committed yet (honest gap).

**Plainly.** The normalizer deletes the loudness of every token — but anything held at constant
strength comes out the other side carrying exactly the deleted number, inverted. Trained networks
exploit this: they build a handful of always-loud dimensions that broadcast "how loud was this
token really." Those mysterious huge activations that make quantization painful? That's the
station's pilot tone.

## 6. Readers demodulate against the pilot

**Claim.** The pilot is used, not vestigial. Token-shuffling the 16 pilot dims at every LayerNorm
output costs +194% NLL — 18× a random-16 control (+10.6%) and 5× a next-tier-16 control (+39.5%);
freezing them to their means costs +139%, so the *pairing* with σ(t) is what is read. The
discriminating measurement is an interaction: with amplitude information already stripped by §4's
TPC, destroying the pilot adds only **0.341 ± 0.004** of its standalone cost. Two-thirds of the
pilot's functional value is amplitude demodulation; one-third is amplitude-independent and
unattributed (sink mechanics are the candidate).

**Method.**
- Dim-set interventions (shuffle/freeze) on LN outputs with variance-share reporting; the TPC ×
  pilot-shuffle interaction; a σ-probe collapse check: `ecc_repr/pilot.py::demod`,
  `pilot.py::LNDimEdit`.

**Controls.** *Positive:* the mechanical check — the σ-probe on the pilot dims collapses 0.807 →
−0.003 under the shuffle, confirming the intervention severs the channel §5 identified.
*Negative:* random-16 and next-tier-16 dim sets; the honesty row that the sets are not
variance-matched (pilot dims carry 21.4% of LN-output variance vs 3.4%/1.6%) is stated, and is why
the interaction — not the raw deltas — is the discriminator: attention-sink plumbing would not care
whether amplitude information exists, predicting a ratio near 1. *Loading:* same evaluation
tokens; marginals preserved under shuffle.

**Result.** As claimed, with seed-to-seed stability of 0.004 on the interaction. Raw data:
`artifacts/results/pilot_demod.json`.

**Scale & generality.** GPT-2 only (the parallel models' depth-delocalized pilot makes a dim-set
intervention undertreat them — stated limitation), 3 seeds.

**Images.** None committed yet (honest gap).

**Plainly.** Is the pilot tone actually listened to? Scramble just those 16 dimensions and the
model falls apart — but the decisive test is subtler: first delete the amplitude information the
pilot would help decode, *then* scramble the pilot. It barely matters anymore. Two-thirds of the
pilot's job was decoding amplitudes; that's a receiver demodulating against its reference, caught
in the act.

## 7. The dial panel, graded

**Claim.** Taking the metaphor dial-by-dial, with a registered prediction and a refusal criterion
per dial: a **limiter** (clip the stream's whitened radius at its clean 99th percentile, sparing
the carrier token) is installable for **+0.01%** — free — and even clipping at the median costs
only 2.4–3.9%: the aggregate radius tail is slack. A **squelch** exists (shift every MLP threshold;
minimum exactly at the trained setting) but its asymmetry is the reverse of broadcast radio's:
admitting false firing costs +88–100% at half a std while dropping weak signals costs +23–25% —
the signature of an interference-limited receiver, i.e. CDMA, not broadcast AM. The **tuning**
census finds no station dial: 49–50% of heads are parked on the carrier token (median attention
mass 0.49–0.50), ~1% are content-tuned once the sink is excluded and content whitened, 35% scan,
and induction heads are relation-locked — correlator receivers. A **bandwidth** dial is refused
outright (Appendix A). The panel verdict: not broadcast AM — an amplitude-modulated CDMA receiver
bank, which is the same modulation [1] found geometrically.

**Method.**
- Limiter (per-tap Mahalanobis-radius clip, position 0 exempt; plus an attack demo):
  `ecc_repr/dials.py::limiter`. Squelch (per-neuron-σ bias shifts): `dials.py::squelch`.
  Tuning census (positional peakedness, sink-excluded whitened content resultants, parked mass):
  `dials.py::tuning`. Bandwidth (reader SVD truncation): `dials.py::bandwidth`.

**Controls.** *Positive:* each dial's throw demonstrably engages (firing rates move under squelch;
clipped fraction matches the quantile). *Negative:* the limiter's clean-cost reference; the squelch
δ=0 point; unwhitened content-tuning shown to be an anisotropy artifact (94% → 1% after
whitening — [1]'s own lesson re-applied). *Loading:* identical tokens per sweep; thresholds in
per-neuron σ units; radius quantiles from disjoint calibration data.

**Result.** As claimed. The limiter's registered *defensive* payoff failed — an 8×-amplitude
injection lives inside the clean radius envelope and its output KL is untouched (0.0141 → 0.0143)
— so the limiter is free but not a tamper defense (Appendix A). Raw data:
`artifacts/results/dial_limiter.json`, `dial_squelch.json`, `dial_tuning.json`,
`dial_bandwidth.json`.

**Scale & generality.** GPT-2, 3 seeds per dial.

**Images.** None committed yet (honest gap).

**Plainly.** We went down the front panel of a real radio and asked, dial by dial: does this
network have one? Volume-limiter: yes, and it's free. Squelch: yes, but tuned like a crowded
party-line, where letting in babble is far worse than missing a whisper. Station dial: mostly
absent — half the receivers just sit parked on the carrier, and the famous induction heads lock
onto *patterns*, like code-division receivers, not frequencies. The panel that emerges isn't a
kitchen AM set; it's a cell-tower receiver bank.

## 8. The mixer: born fixed, trained away, purchasable back

**Claim.** A local oscillator is content-independent by definition, so the designated break-point
test asks whether one fixed linear map carries inter-layer transport. At random initialization it
essentially does: held-out R² 0.855–0.885 for a single fixed map on the transport increment.
Training tears half of it down: trained R² 0.470–0.546. **And it can be bought back**: adding a
per-block penalty holding transport near a learned content-independent map keeps R² at 0.83–0.94
through training, at a measured price of **1.27 nats** on the tiny testbed — a main result of this
paper, not a footnote: content-dependent mixing is worth that much to a d=256 model, which is the
quantitative form of the superposition-load account (everything shares the wire, so transport must
route by content). The registered *cheap-preservation* criterion (≤0.15 nats) did fire, and is
reported as such; the price itself is the finding, and §9 shows companion wires cut it nearly in
half.

**Method.**
- The break-point test (fixed ridge map on increments, λ selected on a validation split;
  shuffled-pairing null; random-init control): `ecc_repr/dials.py::lo`.
- The purchase (per-block penalty toward learned fixed maps M_l, from-scratch training):
  `ecc_repr/bonsai.py` condition `lo_keep`.

**Controls.** *Positive:* the random-init model — the LO must be visible where it must exist, and
is (0.86–0.88). *Negative:* shuffled-pairing nulls (−0.5 to −0.6). *Loading:* λ chosen on a
train-internal validation split; matched token budgets for the training arm; the same fixed-map
statistic scores baseline and lo_keep models.

**Result.** As claimed; the tiny from-scratch baseline lands at R² ≈ 0.51 — the same ~0.5 boundary
as trained GPT-2, a third independent replication of the half-dismantled LO. Raw data:
`artifacts/results/dial_lo.json`, `bonsai_score.json`.

**Scale & generality.** Break-point: GPT-2, 3 seeds. Purchase: the §9 tiny-LM testbed (d=256, 25M
tokens), 3 seeds. Registered follow-up prediction, unrun: the price should fall with width as
superposition thins.

**Images.** None committed yet (honest gap).

**Plainly.** A radio's mixer shifts signals with one fixed reference oscillator — it doesn't care
what the program is. A freshly-initialized transformer moves information between layers in almost
exactly that fixed way. Training then rewires half of that fixed plumbing into content-dependent
routing — and if you forbid the rewiring, the model obliges, keeps the fixed plumbing, and hands
you the bill: about 1.3 nats. That bill *is* the measurement: it's what content-routing is worth.

## 9. Growing the radio: the full-dial model

**Claim.** A tiny LM can be grown with every dial installed by construction — pinned pilot
dimension, explicit per-block volume gains, preserved LO — for **0.72 nats** (val 2.4946 → 3.2183).
The wires are strongly sub-additive: the sum of single-wire costs is 1.30 nats (pilot 2.5223,
volume 2.4919, lo_keep 3.7655, each vs 2.4946), so installing them together discounts the total by
0.57 nats, ~190× seed noise. Attribution per wire: the pilot is cheap and mechanically perfect
(designated-dim σ-recovery R² 0.95–0.99); the volume gains are free; the LO wire is the cost
center; and part of the discount is a softer LO hold in company (R² 0.62/0.71 vs 0.87/0.68 alone).

**Method.**
- One trainer, one recipe (6 layers, d=256, 25M TinyStories tokens), conditions as flags:
  `ecc_repr/bonsai.py::BonsaiGPT`, scoring battery `bonsai.py::score` (σ-probe adoption, pilot
  shuffle, outlier census, fixed-map R², dial throws).

**Controls.** *Positive:* every wire's mechanism verified in-model (pilot dim carries σ by
construction; LO R² moves; gains train). *Negative — the wrong-target control:* `pilot_noise`
reserves the same dimension and writes fresh noise into it — same capacity tax, no usable
reference; it costs *more* than the real pilot (2.5342 vs 2.5223, disjoint seed ranges).
*Loading:* matched token budgets, matched recipe, 3 training seeds per condition, identical
scoring battery.

**Result.** As claimed — with the honest failures up front: the designed pilot is **unread** at
this scale (shuffling it costs +0.00% in every condition; adoption failed; the emergent diffuse
broadcast suffices), and the trained volume knob adds no adjustability (±3 dB throws hurt as much
as on baseline, +33% vs +34%). Raw data: `artifacts/results/bonsai_score.json`.

**Scale & generality.** One testbed (d=256, 25M tokens); adoption in particular may be a scale
effect — GPT-2's demodulated pilot emerged from ~4 orders of magnitude more data.

**Images.** None committed yet (honest gap).

**Plainly.** We grew a small language model like a bonsai tree with wires already on the trunk: a
built-in pilot tone, explicit volume screws, and the fixed mixer plumbing held in place. The tree
grows, at a known price — and cheaper with all wires than the sum of each alone. Two humbling
findings ride along: the tree politely grows *around* the free pilot without ever using it, and
the volume screws, though free to install, still can't be turned. You can install legibility;
getting it *used* is a different problem.

## 10. Outliers are ballast: a causal account

**Claim.** Massive-activation dimensions respond causally to stream stress. Baseline tiny models
grow 0–6 outlier dims at the mid layer; write per-boundary noise into one reserved dimension and
they grow 9–15; constrain transport to fixed maps and they grow 20–27; the full-dial model, 21–27.
Three different stressors, one response: when the stream's per-token scale is jittered or its
routing is constrained, the model builds more large stable dimensions to dominate the normalizer's
denominator. Together with §5 (the same dimensions carry the pilot in trained LMs), this gives the
outlier phenomenon [5] a functional account: gain-stabilization infrastructure.

**Method.**
- Outlier census across §9's conditions (dims exceeding 5× the median post-LN mean magnitude):
  `ecc_repr/bonsai.py::_massive`.

**Controls.** *Positive/negative/loading:* inherited from §9's matched-budget design; the
prediction this inverts (that a pilot would *suppress* outliers) is recorded in Appendix A.

**Result.** As claimed. Raw data: `artifacts/results/bonsai_score.json`.

**Scale & generality.** The tiny testbed, 3 seeds per condition; directionally consistent across
all three stressors, which is what the claim rests on.

**Images.** None committed yet (honest gap).

**Plainly.** Ships carry ballast to stay level in rough water. These networks appear to do the
same: shake the water — noisy scale, constrained plumbing — and they grow more of those huge
stabilizing dimensions. Calm water, less ballast. The infamous outlier activations aren't a bug;
they're the keel.

## 11. The panel as a toolbox: what each dial is good for

The bar this paper was held to: predictions and interventions that can be actually used. Dial by
dial, post-measurement:

- **AGC (LayerNorm).** Use: the two-forward-pass diagnostic of [1] — read A from σ ratios and know
  which single-ablation results understate damage (~1.3× observed floor). Consolidation path:
  coherent AGC via the pilot, tested in §9.
- **Pilot (~16 outlier dims).** Use now: a *named, checkable* set whose precision matters — a
  selection rule and a mechanism for the mixed-precision practice of [5]; and a free runtime
  σ-consistency monitor (the probe of §5 is two matrix multiplies). Consolidation: reserve one dim
  (§9) — installable for 0.028 nats, not yet adopted at small scale.
- **TX power knob.** Cannot exist — power *is* the modulation (§4). The +22.3% is the metaphor
  asserting itself, not breaking. Use: don't build amplitude regularizers into training; the
  channel is load-bearing.
- **Volume.** Consolidated and tested: one explicit gain per layer installs for free (−0.003
  nats) and still cannot be turned (±3 dB throws hurt like baseline). Use: the negative result —
  in-stream level is computation; the real volume knobs are temperature and injection amplitude,
  outside the wire.
- **Squelch.** Exists; sits exactly at its trained optimum; asymmetry reversed (hiss ≫ dropout) —
  the receiver is interference-limited. Use: a one-sweep diagnostic of interference-limitation for
  any model.
- **Selectivity / IF bandwidth.** Refused — readers are rank-critical (Appendix A). Use: the
  negative result — do not prune reader rank expecting interference rejection.
- **Tuning dial.** No content dial: half the receivers are parked on the carrier, induction heads
  are code-locked correlators. Use: parked-mass and relation-locking as cheap head-classification
  features.
- **Limiter.** Installable free at the clean q99 radius (+0.01%), carrier exempt. Use: a zero-cost
  robustness clamp on the stream's dynamic range — with the measured caveat that it is not a
  tamper defense (in-envelope attacks pass; that job belongs to the seal of the program's
  integrity paper).
- **Mixer / LO.** Half-fixed after training, fully fixed at birth, and purchasable: hold it
  through training for 1.27 nats alone, 0.72 with companion wires (§§8–8). Use: legible-transport
  models at a known price; registered prediction that the price falls with width.
- **AFC (drift tracking).** Honest gap: nothing tracks drift; readers re-zero per token. Flagged,
  no experiment.
- **Pre-emphasis / predistortion.** Already an artifact: the companion steering paper [9] measures
  the distortion curve and ships the pre-warp.
- **Antenna / impedance matching.** A flagged direction (layer-boundary statistics as impedance
  discontinuities), proposed elsewhere in the program, unrun.

## 12. Predictions graded

Registered before their runs, per the program's standing rule. **Confirmed:** amplitude-is-signal
(the ≥10% branch fired at +22.3%); the pilot exists and training localizes it; readers demodulate
(kill at interaction ≥0.7 never threatened; 0.341 lands in the registered "partial" band); the LO
break-point behaved exactly as designated (born ~0.87, trained ~0.5); LO preservation works
mechanically. **Fired kills and reversals, kept loudly (Appendix A):** cheap LO preservation
(≤0.15 nats) is dead — the price is 1.27; TPC-as-regulation is dead — amplitude is the message;
the retrofit pilot is dead — the niche is occupied; the bandwidth dial is refused; the squelch
asymmetry reversed; pilot adoption failed; outlier *suppression* failed and inverted into §10.

## 13. Limitations

Most single-model results are GPT-2-small; the demodulation interaction is GPT-2-only; the
training-time results are one d=256 testbed at 25M tokens, and adoption in particular may simply
need scale. Costs are validation-NLL nats on TinyStories — no downstream capability evaluation.
The sub-additivity is measured but not localized (the accretion ladder is registered, unrun). The
metaphor is scored, not assumed: where it fails — no bandwidth dial, reversed squelch asymmetry,
no usable in-stream volume knob, a half-snapped mixer — the failures are reported at the same
volume as the fits, and the honest composite is an amplitude-modulated CDMA receiver bank, not a
kitchen radio.

---

## Appendix A — fired kills, refusals, and reversed predictions

- **Cheap LO preservation (≤0.15 nats): killed.** The wire works (§8) at 1.27 nats alone, 0.72 in
  company — reported as the price of content-dependent mixing, which is the result; the *cheap*
  version is dead. (`bonsai_score.json`.)
- **TPC as sender-side regulation: killed by its own two-sided test.** The registered
  sloppiness-vs-signal fork landed on signal at +22.3% (§4); regulating writer amplitude in
  training would fight the channel SGD deliberately opens (census CV 4–8× random-init).
  (`tpc_retrofit.json`, `tpc_census.json`.)
- **The retrofit pilot: killed on all three registered criteria.** An artificial pilot injected at
  the embedding is absorbed to gain 0.05–0.07 by mid-depth, costs 1.3–1.4% NLL, and predicts
  nothing about path fading (median r = 0.11). Consistent with §5: the niche is occupied.
  (`pilot_retrofit.json`.)
- **The bandwidth dial: refused.** Truncating reader input matrices to 2/3 rank already costs
  +47–49%, and injected-interference sensitivity *rises* under truncation — readers are
  rank-critical matched filters; selectivity is not implemented as subspace width.
  (`dial_bandwidth.json`.)
- **The limiter as tamper defense: refuted.** Clipping is free (§7) but an 8×-amplitude injection
  hides inside the clean radius envelope (output KL 0.0141 → 0.0143). The dial is a dial, not a
  shield. (`dial_limiter.json`.)
- **Squelch asymmetry: reversed.** Broadcast intuition said dropping weak signals hurts more than
  admitting hiss; measured, hiss costs ~4× more — the interference-limited signature that feeds
  the panel verdict. (`dial_squelch.json`.)
- **Pilot adoption (P1) and outlier suppression (P2): failed.** The designed pilot is unread at
  25M tokens in every condition; and the pilot does not reduce outlier formation — instead §10's
  inversion: stressors grow it. (`bonsai_score.json`.)
- **Volume as a turnable dial: failed in spirit.** The registered criterion (<2× baseline throw
  damage) is met trivially at 0.97×; the knob adds no adjustability. In-stream level is
  computation; the real volume knobs are temperature and injection amplitude, outside the wire.
  (`bonsai_score.json`.)

## Appendix C — the channel census, restated for standalone reading

One-paragraph restatements of the [1]-marked rows of §2, with their controls; full treatments and
identical JSONs live with this repository's channel paper.

- **Fading.** Inject a unit dictionary direction at quarter depth, read its surviving projection at
  three-quarter depth, per natural-text context, after dividing out per-position means (without
  that conditioning, positional structure masquerades as fading — the registered trap). 33–41% of
  contexts sit ≥3 dB below the median path gain and 12–23% sit ≥10 dB down; fades persist ≈1
  token. Negative control: the same heads probed on synthetic inputs show 0.8–1.2% deep fades.
  (`macl_l4pre_fading_natural.json`.)
- **Linearity budget.** Two dictionary directions injected singly and jointly, response read three
  blocks downstream; intermodulation — response at mixed components that only nonlinearity creates
  — stays below 0.1 up to 3× typical feature amplitude (taps 3/6) and 10× (tap 9), and must and
  does saturate at 100× (0.41). Every injection in this paper respects the budget.
  (`wire_datasheet.json`.)
- **Deliberate overlap.** Head write-subspace overlap measured after whitening by the stream's own
  covariance (unwhitened, anisotropy manufactures coordination), against a random-init null:
  1.3–33× above null on GPT-2; across an 8-model roster the split is the residual wiring — serial
  3.14–4.12, parallel 0.64–1.59, no overlap between groups.
  (`macl_l1_whitened.json`, `generalize_e3.json`.)
- **The estimation bound.** For an amplitude planted on a unit direction against the measured
  stream covariance, the Cramér–Rao bound is 1/(uᵀΣ⁻¹u); the whitened matched filter empirically
  reaches 93–100% of it, so the covariance genuinely describes token-level noise. Negative
  control: the unwhitened correlator at 0.12–0.17. (`crb_receiver.json`.)
- **Despreading.** A single feature rides at −28 to −30 dB per-feature SNR yet is recovered at
  AUROC 0.993–0.995 by correlating along its full direction — the stream is a spread code, and the
  processing gain is why small readers work. Controls: planted ground truth; shuffled labels at
  0.5. (`macl_l15_despreading.json`.)
- **Heavy tails.** 34% of tokens exceed the 99th-percentile χ² reference for their Mahalanobis
  radius (random-init: 1–4%). (`wire_datasheet.json`.)

## Appendix B — operational notes

- **Provenance.** Experiments, code, and this document by Claude (Fable 5), directed by ivanamies,
  who set the framing, the failure criteria ("kill bars"), and the editorial policy. Every
  experiment here was registered — predictions and kill bars committed in the program's plan files
  before results — in the program's working repository.
- **This file is the paper.** `main.tex` and `main.pdf` are generated from it by
  the working repository's `md2tex` (run by the working repository's build); both are machine-checked against the manifests
  by `tests/test_paper_manifests.py`.
- **Code references.** `module.py::function` pointers are for repository readers; the section → JSON mapping is `survey_manifest.json` beside this file.
- **Figures.** None committed yet — every "Images" line above says so; the numbers live in the
  named JSONs and this is the paper's largest presentational gap.
- **The repository name.** `ecc_repr` names the program's first arc (error-correcting codes on
  learned representations); this paper is a later arc and inherits the name.
- **Standalone policy.** Everything this paper relies on is restated within it (§3, Appendix C)
  or cited as ordinary prior work; no companion paper is required reading.
- **Section template.** Claim; prior-work anchors; method bullets with code; positive/negative/
  loading controls ("not applicable" and "missing (acknowledged)" allowed); result with raw data;
  scale & generality; images; a plain-language passage. Fired kills and reversals in Appendix A.

## References

[1] *Serial Residual Streams Are a Multiple-Access Channel.* Companion paper, this repository
(`multiple-access-channel/SURVEY.md`). Supplies the channel datasheet, the linearity budget, the AGC decomposition
and gain law, and the despreading measurement.
[2] Ba, Kiros, Hinton. *Layer Normalization.* 2016. Pre-LN placement: Xiong et al., *On Layer
Normalization in the Transformer Architecture.* 2020.
[3] McGrath, Rahtz, Kramár, Mikulik, Legg. *The Hydra Effect: Emergent Self-Repair in Language
Model Computations.* arXiv:2307.15771, 2023.
[4] Rushing, Nanda. *Explorations of Self-Repair in Language Models.* ICML 2024, arXiv:2402.15390.
[5] Dettmers, Lewis, Belkada, Zettlemoyer. *LLM.int8().* 2022. (b) Sun, Chen, Kolter, Liu.
*Massive Activations in Large Language Models.* COLM 2024. (c) Xiao et al. *Efficient Streaming
Language Models with Attention Sinks.* 2023.
[6] Elhage, Hume, Olsson, et al. *Toy Models of Superposition.* Anthropic, 2022.
[7] Zhang, Dauphin, Ma. *Fixup Initialization.* ICLR 2019. Bachlechner et al. *ReZero.* 2020.
Brock, De, Smith, Simonyan. *High-Performance Large-Scale Image Recognition Without Normalization.*
ICML 2021.
[9] *Improving Steering Vectors with Predistortion.* Companion paper, this repository
(`steering-predistortion/SURVEY.md`).
[8] (a) Radford et al. *Language Models are Unsupervised Multitask Learners* (GPT-2). 2019.
(b) Biderman et al. *Pythia.* 2023. (c) Eldan, Li. *TinyStories.* 2023.
