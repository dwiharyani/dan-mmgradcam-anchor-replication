# Method

The experiment asks whether 1D and 2D Grad-CAM maps for the same ECG remain
spatially consistent under mild perturbations of the raw Lead-II signal.

1. Generate 30 perturbed versions of each raw ECG:
   - 6 amplitude scales: 0.8, 0.9, 0.95, 1.05, 1.1, 1.2.
   - 12 baseline-drift signals: 0.3 Hz sine wave, 5% and 10% of record
     peak-to-peak amplitude, 6 phases per level.
   - 12 Gaussian-noise draws: 5% and 10% of record standard deviation,
     6 seeds per level.
2. Regenerate both model inputs from the same perturbed signal: the 1D tensor
   using the 1D scaler and a re-rendered 2D image.
3. Run the fixed pretrained 1D and 2D models; models are not retrained.
   Grad-CAM targets each branch's own original prediction.
4. Retain a perturbation for a branch only when that branch preserves its own
   original prediction. Report prediction-retention rate (PRR) separately.
5. On a shared 600-column grid, compute mean attribution `mu_b`, standard
   deviation `sigma_b`, stability `S_b = 1 - sigma_b/(max(sigma_b)+eps)`,
   reliability-weighted attribution `H_b = mu_b*S_b`, and fused map
   `R = sqrt(H_1D*H_2D)`. A record with fewer than two accepted maps is
   unevaluable.

Acceptance is branch-specific; the shared accepted set is used when calculating
the fused R map. See the notebook and the supplied method note for implementation
details.
