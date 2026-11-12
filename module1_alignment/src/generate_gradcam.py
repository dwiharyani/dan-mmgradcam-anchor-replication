from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path.home() / "Alignment-XAI"

MODEL_1D = ROOT / "models/MMGradCAM_1d_trained_model.keras"
MODEL_2D = ROOT / "models/MMGradCAM_2d_trained_model.keras"
SCALER = ROOT / "models/preprocessing/standard_scaler.npz"

PILOT = ROOT / "module1_alignment/data/pilot_manifest.csv"
ECG_ROOT = Path.home() / "XAI/1D-ECG-Signal"
IMAGE_ROOT = Path.home() / "XAI/imageData-Label2/label1/Testdata"

OUT = ROOT / "module1_alignment/results/gradcam_pilot"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["AFib", "GSVT", "SB", "SR"]


def find_last_conv(model, layer_type):
    for layer in reversed(model.layers):
        if isinstance(layer, layer_type):
            return layer
    raise ValueError("Convolutional layer not found")


def gradcam(model, x, conv_layer, target_idx):
    grad_model = tf.keras.Model(
        model.inputs,
        [conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        features, predictions = grad_model(x, training=False)
        target_score = predictions[:, target_idx]

    gradients = tape.gradient(target_score, features)

    if gradients is None:
        raise RuntimeError("Grad-CAM gradients unavailable")

    if len(features.shape) == 3:
        weights = tf.reduce_mean(gradients, axis=1)
        cam = tf.reduce_sum(
            features * weights[:, None, :], axis=-1
        )
        cam = cam[0]

    elif len(features.shape) == 4:
        weights = tf.reduce_mean(gradients, axis=(1, 2))
        cam = tf.reduce_sum(
            features * weights[:, None, None, :], axis=-1
        )
        cam = cam[0]

    else:
        raise ValueError("Unexpected feature shape")

    cam = tf.nn.relu(cam)

    maximum = tf.reduce_max(cam)
    cam = cam / (maximum + 1e-8)

    return cam


print("Loading pretrained models...", flush=True)

model1 = tf.keras.models.load_model(MODEL_1D, compile=False)
model2 = tf.keras.models.load_model(MODEL_2D, compile=False)

conv1 = find_last_conv(model1, tf.keras.layers.Conv1D)
conv2 = find_last_conv(model2, tf.keras.layers.Conv2D)

scaler = np.load(SCALER, allow_pickle=False)
mean = scaler["mean"]
scale = scaler["scale"]

pilot = pd.read_csv(PILOT)

maps1 = []
maps2 = []
metadata = []

for _, row in pilot.iterrows():

    target_idx = CLASSES.index(row["class_name"])

    # 1D ECG
    ecg = pd.read_csv(ECG_ROOT / row["ecg_csv_filename"])

    signal = ecg["II"].to_numpy(dtype=np.float32)
    assert signal.shape == (5000,)
    assert np.isfinite(signal).all()

    signal = ((signal - mean) / scale).astype(np.float32)
    x1 = tf.convert_to_tensor(signal.reshape(1, 5000, 1))

    # 2D ECG
    image_path = IMAGE_ROOT / row["image_relative_path"]

    image = tf.io.read_file(str(image_path))
    image = tf.io.decode_png(image, channels=1)
    image = tf.image.resize(image, (300, 600))
    image = tf.cast(image, tf.float32) / 255.0
    x2 = tf.expand_dims(image, axis=0)

    # Generate explanations for the SAME target class
    cam1 = gradcam(model1, x1, conv1, target_idx)
    cam2 = gradcam(model2, x2, conv2, target_idx)

    cam1 = tf.image.resize(
        cam1[None, :, None, None],
        (5000, 1),
        method="bilinear"
    )[0, :, 0, 0].numpy()

    cam2 = tf.image.resize(
        cam2[None, :, :, None],
        (300, 600),
        method="bilinear"
    )[0, :, :, 0].numpy()

    maps1.append(cam1.astype(np.float32))
    maps2.append(cam2.astype(np.float32))

    metadata.append({
        "cohort_index": int(row["cohort_index"]),
        "class_name": row["class_name"],
        "target_idx": target_idx,
        "ecg_csv_filename": row["ecg_csv_filename"],
        "image_relative_path": row["image_relative_path"],
    })

    print(
        f"Completed: {row['class_name']} "
        f"(cohort {row['cohort_index']})",
        flush=True
    )

maps1 = np.stack(maps1)
maps2 = np.stack(maps2)

assert maps1.shape == (4, 5000)
assert maps2.shape == (4, 300, 600)
assert np.isfinite(maps1).all()
assert np.isfinite(maps2).all()

np.save(OUT / "gradcam_1d.npy", maps1)
np.save(OUT / "gradcam_2d.npy", maps2)

pd.DataFrame(metadata).to_csv(
    OUT / "gradcam_metadata.csv", index=False
)

print("\nSaved Grad-CAM results:")
print("1D shape:", maps1.shape)
print("2D shape:", maps2.shape)
print("Output:", OUT)
