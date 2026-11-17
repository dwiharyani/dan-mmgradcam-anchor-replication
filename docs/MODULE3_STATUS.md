# DAN-MMGradCAM — Module 3

## Module 3A: Reliability-Weighted Attribution

R(u) = sqrt(H1j(u) * H2j(u))

Dataset version: Module 2 s9534
Module 1 frozen primary cohort: 727 ECGs
Module 3A eligible: 694 ECGs
Matched 30/30 controlled ablation: 551 ECGs

Status: Implemented and numerically verified.

## Module 3B: Waveform-Conditioned Support

R_trace(u) = R(u) * F_trace(u)^lambda

Pilot: ECG 209 (AFib)
Tested lambda values: 0.5, 1.0, 2.0

Status: Single-record pilot completed.

## Important limitations

- Module 2 uses rerendered ECG images and an on-trace projection.
- This is not identical to the Module 1 G12 projection.
- Full-cohort Module 3B evaluation is pending.
- Faithfulness, sanity checks, and statistical validation are pending.
- No improvement over baseline is claimed yet.
