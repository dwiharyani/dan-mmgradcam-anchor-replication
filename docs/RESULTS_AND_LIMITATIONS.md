# Results and limitations

## Exploratory full-v2 run

- Records: 1,054
- Perturbations per record: 30
- Median 1D stability: 0.8197
- Median 2D stability: 0.4687
- Median fused reliability-weighted map score R: 0.0747
- Mean 1D prediction-retention rate: 0.9744
- Mean 2D prediction-retention rate: 0.9758
- Correlation between 1D and 2D stability: 0.221
- Branch disagreement rate: 0.0939
- 1D test accuracy in this run: 0.9469
- 2D accuracy on re-rendered images: 0.925

All 1,054 records were evaluable for the 1D, 2D, and joint analyses in this run.

## Interpretation

The 1D branch has a higher median stability score in this exploratory run.
Prediction-retention rates are high for both branches. Stability measures
consistency under the selected perturbations; it is not, by itself, a measure
of faithfulness, causal importance, or clinical relevance.

## Reproducibility caveats

1. The original `standard_scaler.npz` was unavailable. The scaler was recomputed
   from 9,592 non-test records, whereas the original cohort contained 9,534.
   The run's 1D test accuracy (0.9469) differs from the repository value (0.9478).
2. The 2D inputs were re-rendered from raw Lead-II ECG using inferred rendering
   geometry. Accuracy on re-rendered images (0.925) is lower than on real images
   (0.9668). Therefore, 2D stability results apply to the re-rendered inputs.
3. Perturbation magnitudes and count have not been validated on the validation split.
4. Exploratory stability and deletion analyses do not establish faithfulness.

Consult the aggregate JSON and CSV files for the numeric outputs available in
this package. Figure captions and notebook cells should be checked against the
final paper narrative before publication.
