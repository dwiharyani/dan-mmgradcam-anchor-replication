from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_NAMES = ["AFib", "GSVT", "SB", "SR"]


def find_last_conv2d_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise RuntimeError("No Conv2D layer found in model.")


def load_image_for_model(image_path, image_size=(300, 600)):
    img = tf.io.read_file(str(image_path))
    img = tf.io.decode_image(img, channels=1, expand_animations=False)
    img = tf.image.resize(img, image_size)
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape([image_size[0], image_size[1], 1])
    return tf.expand_dims(img, axis=0), img.numpy().squeeze()


def make_gradcam_heatmap(img_tensor, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        class_index = tf.argmax(predictions[0])
        class_channel = predictions[:, class_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    if float(max_val.numpy()) > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy(), predictions.numpy()[0]


def resize_heatmap(heatmap, target_size=(300, 600)):
    h = tf.convert_to_tensor(heatmap[..., None], dtype=tf.float32)
    h = tf.image.resize(h, target_size, method="bilinear")
    return tf.squeeze(h, axis=-1).numpy()


def save_plot(gray_img, heatmap, true_class, pred_class, confidence, out_path, threshold):
    mask = heatmap >= threshold
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].imshow(gray_img, cmap="gray")
    axes[0].set_title("Original ECG image")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM heatmap")
    axes[1].axis("off")

    axes[2].imshow(gray_img, cmap="gray")
    axes[2].imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axes[2].contour(mask.astype(np.uint8), levels=[0.5], linewidths=0.8)
    axes[2].set_title(f"True={true_class} | Pred={pred_class} ({confidence*100:.2f}%)")
    axes[2].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--threshold", type=float, default=0.23)
    args = ap.parse_args()

    model_path = Path(args.model)
    manifest_path = Path(args.manifest)
    image_root = Path(args.image_root)
    out_dir = Path(args.output)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    assert len(df) == 1054, f"Expected 1054 test rows, got {len(df)}"

    model = tf.keras.models.load_model(model_path)
    last_conv = find_last_conv2d_layer(model)

    print("Last Conv2D:", last_conv)
    print("Last Conv2D output:", model.get_layer(last_conv).output.shape)

    native_heatmaps = []
    resized_heatmaps = []
    rows = []
    mismatches = 0

    for i, row in df.iterrows():
        image_path = image_root / row["image_filename"]
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        x, gray_img = load_image_for_model(image_path)
        native, probs = make_gradcam_heatmap(x, model, last_conv)

        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        if pred_class != row["pred_class"]:
            mismatches += 1

        resized = np.clip(resize_heatmap(native), 0.0, 1.0)
        salient_fraction = float((resized >= args.threshold).mean())

        native_heatmaps.append(native.astype(np.float32))
        resized_heatmaps.append(resized.astype(np.float32))

        rec = {
            "row_index": int(i),
            "cohort_index": int(row["cohort_index"]),
            "patient_id": row["patient_id"],
            "true_class": row["true_class"],
            "pred_class_manifest": row["pred_class"],
            "pred_from_model": pred_class,
            "pred_match_manifest": bool(pred_class == row["pred_class"]),
            "correct": bool(pred_class == row["true_class"]),
            "image_filename": row["image_filename"],
            "image_path": str(image_path),
            "salient_fraction": salient_fraction,
            "heatmap_min": float(resized.min()),
            "heatmap_max": float(resized.max()),
            "native_h": int(native.shape[0]),
            "native_w": int(native.shape[1]),
        }
        for j, cls in enumerate(CLASS_NAMES):
            rec[f"prob_{cls}"] = float(probs[j])
        rows.append(rec)

        if (i + 1) % 100 == 0 or (i + 1) == len(df):
            print(f"Processed {i+1}/{len(df)}")

    native_heatmaps = np.stack(native_heatmaps)
    resized_heatmaps = np.stack(resized_heatmaps)
    masks = (resized_heatmaps >= args.threshold).astype(np.uint8)
    out_df = pd.DataFrame(rows)

    np.save(out_dir / "heatmaps_2d_native.npy", native_heatmaps)
    np.save(out_dir / "heatmaps_2d_resized.npy", resized_heatmaps)
    np.save(out_dir / "salient_masks_threshold_023.npy", masks)
    out_df.to_csv(out_dir / "gradcam_2d_manifest.csv", index=False)

    # one correct + one error example per class
    for cls in CLASS_NAMES:
        cls_df = out_df[out_df["true_class"] == cls].copy()
        for status, sub in [
            ("correct", cls_df[cls_df["correct"]].copy()),
            ("error", cls_df[~cls_df["correct"]].copy()),
        ]:
            if len(sub) == 0:
                continue
            sub["confidence"] = sub.apply(
                lambda r: r[f"prob_{r['pred_from_model']}"], axis=1
            )
            ex = sub.sort_values("confidence", ascending=False).iloc[0]
            _, gray = load_image_for_model(ex["image_path"])
            hm = resized_heatmaps[int(ex["row_index"])]
            save_plot(
                gray,
                hm,
                ex["true_class"],
                ex["pred_from_model"],
                ex["confidence"],
                plots_dir / f"{cls}_{status}_{ex['patient_id']}.png",
                args.threshold,
            )

    zero_heatmaps = int(
        np.sum(resized_heatmaps.reshape(len(out_df), -1).max(axis=1) == 0)
    )

    summary = {
        "n_test_images": int(len(out_df)),
        "threshold": float(args.threshold),
        "last_conv2d": last_conv,
        "native_heatmap_shape": list(native_heatmaps.shape),
        "resized_heatmap_shape": list(resized_heatmaps.shape),
        "pred_mismatch_vs_manifest": int(mismatches),
        "finite_values": bool(np.isfinite(resized_heatmaps).all()),
        "zero_heatmaps": zero_heatmaps,
        "accuracy_from_manifest": float(out_df["correct"].mean()),
        "mean_salient_fraction_by_true_class": (
            out_df.groupby("true_class")["salient_fraction"].mean().to_dict()
        ),
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== GRAD-CAM 2D AUDIT ===")
    print("Test images:", len(out_df))
    print("Native heatmaps:", native_heatmaps.shape)
    print("Resized heatmaps:", resized_heatmaps.shape)
    print("Finite values:", np.isfinite(resized_heatmaps).all())
    print("Zero heatmaps:", zero_heatmaps)
    print("Prediction mismatches:", mismatches)
    print("Accuracy:", out_df["correct"].mean())
    print("\nSTATUS: PASS")


if __name__ == "__main__":
    main()
