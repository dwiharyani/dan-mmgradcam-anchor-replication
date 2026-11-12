from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path.home() / "Alignment-XAI"

MODEL_PATH = ROOT / "models/MMGradCAM_1d_trained_model.keras"
SCALER_PATH = ROOT / "models/preprocessing/standard_scaler.npz"
TEST_CSV = ROOT / "dan-mmgradcam-anchor-replication/reproducibility/test_split_1054.csv"
ECG_ROOT = Path.home() / "XAI/1D-ECG-Signal"
OUTPUT = ROOT / "module1_alignment/results"

CLASSES = ["AFib", "GSVT", "SB", "SR"]

OUTPUT.mkdir(parents=True, exist_ok=True)

print("Loading original StandardScaler...", flush=True)
scaler = np.load(SCALER_PATH, allow_pickle=False)

mean = scaler["mean"]
scale = scaler["scale"]

assert mean.shape == (5000,)
assert scale.shape == (5000,)
assert np.isfinite(mean).all()
assert np.isfinite(scale).all()
assert (scale > 0).all()

print("Loading pretrained 1D CNN...", flush=True)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

test = pd.read_csv(TEST_CSV)
assert len(test) == 1054

predictions = []

for i, row in test.iterrows():

    ecg_path = ECG_ROOT / row["ecg_csv_filename"]

    if not ecg_path.is_file():
        raise FileNotFoundError(ecg_path)

    ecg = pd.read_csv(ecg_path)

    signal = pd.to_numeric(
        ecg["II"], errors="coerce"
    ).to_numpy(dtype=np.float32)

    assert signal.shape == (5000,)
    assert np.isfinite(signal).all()

    # Apply original training scaler
    signal = ((signal - mean) / scale).astype(np.float32)

    # CNN input: (batch, time, channels)
    x = signal.reshape(1, 5000, 1)

    scores = model(x, training=False).numpy()[0]

    assert scores.shape == (4,)
    assert np.isfinite(scores).all()

    pred_idx = int(np.argmax(scores))

    predictions.append({
        "cohort_index": int(row["cohort_index"]),
        "ecg_csv_filename": row["ecg_csv_filename"],
        "true_class": row["class_name"],
        "predicted_class": CLASSES[pred_idx],
        "confidence": float(scores[pred_idx]),
        "correct": bool(
            CLASSES[pred_idx] == row["class_name"]
        ),
    })

    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(test)}", flush=True)

results = pd.DataFrame(predictions)

accuracy = accuracy_score(
    results["true_class"],
    results["predicted_class"]
)

print("\n" + "=" * 60)
print("1D CNN TEST RESULTS")
print("=" * 60)

print(f"Total samples: {len(results)}")
print(f"Correct predictions: {results['correct'].sum()}")
print(f"Test accuracy: {accuracy:.6f} ({accuracy*100:.2f}%)")

print("\nClassification report:")
print(classification_report(
    results["true_class"],
    results["predicted_class"],
    labels=CLASSES,
    zero_division=0,
))

print("\nConfusion matrix:")
print(confusion_matrix(
    results["true_class"],
    results["predicted_class"],
    labels=CLASSES,
))

output_file = OUTPUT / "1d_test_predictions.csv"
results.to_csv(output_file, index=False)

print(f"\nSaved predictions to: {output_file}")
