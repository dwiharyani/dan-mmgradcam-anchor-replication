from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path.home() / "Alignment-XAI"

MODEL = ROOT / "models/MMGradCAM_2d_trained_model.keras"
TEST_CSV = ROOT / "dan-mmgradcam-anchor-replication/reproducibility/test_split_1054.csv"
IMAGE_ROOT = Path.home() / "XAI/imageData-Label2/label1/Testdata"
OUTPUT = ROOT / "module1_alignment/results"

CLASSES = ["AFib", "GSVT", "SB", "SR"]

OUTPUT.mkdir(parents=True, exist_ok=True)

print("Loading pretrained 2D model...")
model = tf.keras.models.load_model(MODEL, compile=False)

df = pd.read_csv(TEST_CSV)

predictions = []

for i, row in df.iterrows():
    path = IMAGE_ROOT / row["image_relative_path"]

    image = tf.io.read_file(str(path))
    image = tf.io.decode_png(image, channels=1)
    image = tf.image.resize(image, (300, 600))
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=0)

    scores = model(image, training=False).numpy()[0]

    pred_idx = int(np.argmax(scores))

    predictions.append({
        "cohort_index": row["cohort_index"],
        "true_class": row["class_name"],
        "predicted_class": CLASSES[pred_idx],
        "confidence": float(scores[pred_idx]),
    })

    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(df)}")

results = pd.DataFrame(predictions)

accuracy = accuracy_score(
    results["true_class"],
    results["predicted_class"]
)

print("\nTEST ACCURACY:", accuracy)

print("\nCLASSIFICATION REPORT:")
print(classification_report(
    results["true_class"],
    results["predicted_class"],
    labels=CLASSES,
    zero_division=0
))

results.to_csv(
    OUTPUT / "2d_test_predictions.csv",
    index=False
)

print("\nSaved predictions successfully.")
