> **A human disclaimer.** The AI has produced stunning achievements in applied
> mathematics. I found myself between jobs for a weekend, so I decided to join the AI
> research train before I became busy again. I directed Claude Fable to work on an old idea of mine
> from the pre-transformer days, whether DNNs are some form of communication channel,
> in the EE sense. The answer appears to be a conditional yes. Too bad too many years have passed
> and I can no longer fully judge its work, but here it is. — ivanamies

# Results: three papers

## [Serial Residual Streams Are a Multiple-Access Channel](multiple-access-channel/SURVEY.md)

**TL;DR.** We took "the residual stream is a multiple-access channel" literally. Heads run deliberate CDMA (serial archs only). LayerNorm is blind AGC — hold it still and ablation damage grows 2–8×; "self-repair" is half gain compensation, half real redundancy. Steering has a duty cycle.

*(277 characters.)*

## [Improving Steering Vectors with Predistortion](steering-predistortion/SURVEY.md)

**TL;DR.** Steering fails like a satellite amplifier: past a knee the vector arrives full-size, pointing wrong — rotation precedes compression, 100% of directions, every model tested. Measure the curve once per feature, invert it upstream: cosine 0.83→0.93 at 64×. Caches: ~96% memoryless.

*(278 characters.)*

## [A Tamper Seal for Neural Networks](tamper-seal/SURVEY.md)

**TL;DR.** A 213k-param head on a frozen LM — no fine-tune, no LM gradients — makes activation and KV-cache integrity a calibrated check: swaps, replays, transplants at 0.96–0.98; Mahalanobis: chance; FPR 0.000 at W=8; flat 124M→6.9B; ≤1% overhead. Authenticates state, not computation.

*(275 characters.)*
