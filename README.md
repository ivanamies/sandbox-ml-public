> **A human disclaimer.** The AI has produced stunning achievements in applied
> mathematics. I found myself between jobs for a weekend, so I decided to join the AI
> research train before I became busy again. I directed Claude Fable to work on an old idea of mine
> from the pre-transformer days, whether DNNs are some form of communication channel,
> in the EE sense. The answer appears to be a conditional yes. Too bad too many years have passed
> and I can no longer fully judge its work, but here it is. — ivanamies

# Results: three papers

## [Serial Residual Streams Are a Multiple-Access Channel](multiple-access-channel/SURVEY.md)

**TL;DR.** We took "the residual stream is a multiple-access channel" literally. Heads run deliberate CDMA (serial archs only). LayerNorm is blind AGC — hold it still and ablation damage grows 2–8×; "self-repair" is half gain compensation, half real redundancy. Steering has a duty cycle.

## [Improving Steering Vectors with Predistortion](steering-predistortion/SURVEY.md)

**TL;DR.** Steering fails like a satellite amplifier: past a measurable knee the vector arrives at full size pointing the wrong way — rotation precedes compression, for 100% of directions tested, on every model tested. Measure the distortion curve once per feature, invert it upstream: cosine 0.83→0.93 at 64×. Cacheable, because the channel is ~96% memoryless.

## [A Tamper Seal for Neural Networks](tamper-seal/SURVEY.md)

**TL;DR.** A 213k-parameter head bolted onto a frozen LM — no fine-tune, no gradients into the model — makes activation and KV-cache integrity a calibrated runtime check: swapped, replayed, or transplanted state caught at 0.96–0.98 where Mahalanobis sits at chance, windowed false-alarm rate 0.000 at 8-token windows, flat from 124M to 6.9B parameters, ≤1% overhead on a server GPU. It authenticates the state; it does not judge the computation.

---

Every number in each paper is backed by a committed JSON under `artifacts/results/`, mapped
per-section by each paper's manifest. Experiments, code, and these documents by Claude (Fable 5),
directed by ivanamies. This is a curated public snapshot; the program's working repository, its
history, and its remaining papers are not part of it.
