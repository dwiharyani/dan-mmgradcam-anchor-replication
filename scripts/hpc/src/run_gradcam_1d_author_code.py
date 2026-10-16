from pathlib import Path
import argparse
import json

import h5py
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model


CLASS_NAMES = ["AFib", "GSVT", "SB", "SR"]


def gradcam_1d(model, signal):
    """
    Reproduce author's 1D Grad-CAM logic:
    - predicted class
    - last Conv1D layer
    - mean gradient weights across temporal dimension
    - weighted sum
    - ReLU
    - normalize
    - interpolate to original signal length
    """

    signal = np.asarray(signal, dtype=np.float32)

    if signal.ndim == 1:
        signal = signal[:, None]

    input_tensor = tf.convert_to_tensor(
        signal[None, ...],
        dtype=tf.float32
    )

    prediction = model(input_tensor, training=False)

    pred_idx = int(
        tf.argmax(prediction[0]).numpy()
    )

    last_conv_layer = next(
        layer for layer in model.layers[::-1]
        if isinstance(layer, tf.keras.layers.Conv1D)
    )

    grad_model = Model(
        inputs=model.inputs,
        outputs=[
            last_conv_layer.output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_output, predictions = grad_model(
            input_tensor,
            training=False
        )

        loss = predictions[:, pred_idx]

    gradients = tape.gradient(
        loss,
        conv_output
    )

    conv_output = conv_output[0]
    gradients = gradients[0]

    # Exact author logic:
    # mean gradient over temporal dimension
    weights = tf.reduce_mean(
        gradients,
        axis=0
    )

    activation_map = tf.reduce_sum(
        conv_output * weights[None, :],
        axis=-1
    )

    activation_map = tf.nn.relu(
        activation_map
    ).numpy()

    max_value = float(
        np.max(activation_map)
    )

    if max_value > 0:
        activation_map = (
            activation_map / max_value
        )
    else:
        activation_map = np.zeros_like(
            activation_map,
            dtype=np.float32
        )

    original_length = signal.shape[0]

    # Match author's np.interp approach.
    resized = np.interp(
        np.arange(original_length),
        np.linspace(
            0,
            original_length,
            activation_map.shape[0]
        ),
        activation_map
    ).astype(np.float32)

    return (
        resized,
        pred_idx,
        prediction[0].numpy(),
        last_conv_layer.name,
        int(activation_map.shape[0]),
    )


def save_plot(
    raw_signal,
    scaled_signal,
    heatmap,
    patient_id,
    true_class,
    pred_class,
    confidence,
    output_path,
    threshold,
):
    t = np.arange(len(raw_signal))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 7),
        sharex=True
    )

    ax1, ax2 = axes

    ax1.plot(
        t,
        raw_signal,
        linewidth=0.8
    )

    salient = heatmap > threshold

    if salient.any():

        y0 = float(np.min(raw_signal))
        y1 = float(np.max(raw_signal))

        ax1.fill_between(
            t,
            y0,
            y1,
            where=salient,
            alpha=0.20
        )

    ax1.set_title(
        f"{patient_id} | "
        f"True={true_class} | "
        f"Pred={pred_class} "
        f"({confidence:.2%})"
    )

    ax1.set_ylabel("Raw Lead II")

    ax2.plot(
        t,
        heatmap,
        linewidth=0.8
    )

    ax2.fill_between(
        t,
        0,
        heatmap,
        where=salient,
        alpha=0.25
    )

    ax2.axhline(
        threshold,
        linestyle="--",
        linewidth=1
    )

    ax2.set_ylim(
        0,
        1.02
    )

    ax2.set_ylabel("Grad-CAM")
    ax2.set_xlabel("Sample (500 Hz)")

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--h5",
        required=True
    )

    ap.add_argument(
        "--model",
        required=True
    )

    ap.add_argument(
        "--split",
        default="splits/frozen_split_author_code.csv"
    )

    ap.add_argument(
        "--scaler",
        default="results/1d_author_code/standard_scaler.npz"
    )

    ap.add_argument(
        "--output",
        default="results/gradcam_1d_author_code"
    )

    ap.add_argument(
        "--threshold",
        type=float,
        default=0.23
    )

    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(
        parents=True,
        exist_ok=True
    )

    plots_dir = out / "plots"
    plots_dir.mkdir(
        exist_ok=True
    )

    print("=" * 78)
    print("MM-GRADCAM 1D GRAD-CAM — AUTHOR_CODE")
    print("=" * 78)

    print("TensorFlow :", tf.__version__)
    print(
        "GPU        :",
        tf.config.list_physical_devices("GPU")
    )

    print("Threshold  :", args.threshold)

    # ---------------------------------------------------------
    # Load locked ECG
    # ---------------------------------------------------------

    with h5py.File(
        args.h5,
        "r"
    ) as f:

        X_raw = f["X"][:]
        y = f["y"][:]

        patient_ids = f[
            "patient_id"
        ][:]

    patient_ids = np.array([
        x.decode()
        if isinstance(x, bytes)
        else str(x)
        for x in patient_ids
    ])

    assert X_raw.shape == (
        10588,
        5000
    )

    # ---------------------------------------------------------
    # Frozen test patients
    # ---------------------------------------------------------

    split = pd.read_csv(
        args.split
    )

    test = (
        split[
            split["split"] == "test"
        ]
        .sort_values("split_order")
        .reset_index(drop=True)
    )

    assert len(test) == 1054

    test_idx = (
        test["cohort_index"]
        .to_numpy(dtype=int)
    )

    # ---------------------------------------------------------
    # Restore EXACT training scaler
    # ---------------------------------------------------------

    scaler = np.load(
        args.scaler
    )

    mean = scaler["mean"]
    scale = scaler["scale"]

    assert mean.shape == (5000,)
    assert scale.shape == (5000,)

    X_test_scaled = (
        (X_raw[test_idx] - mean)
        / scale
    ).astype(np.float32)

    # ---------------------------------------------------------
    # Load successful trained model
    # ---------------------------------------------------------

    model = tf.keras.models.load_model(
        args.model
    )

    last_conv = next(
        layer for layer in model.layers[::-1]
        if isinstance(
            layer,
            tf.keras.layers.Conv1D
        )
    )

    print("\nModel:")
    print(args.model)

    print(
        "Last Conv1D:",
        last_conv.name
    )

    print(
        "Last Conv1D output:",
        last_conv.output.shape
    )

    # Expected from this architecture:
    # temporal size around 156, channels 128

    # ---------------------------------------------------------
    # Generate all 1054 test heatmaps
    # ---------------------------------------------------------

    heatmaps = np.zeros(
        (len(test), 5000),
        dtype=np.float32
    )

    masks = np.zeros(
        (len(test), 5000),
        dtype=np.uint8
    )

    rows = []

    plot_candidates = []

    for test_order, cohort_idx in enumerate(
        test_idx
    ):

        raw = X_raw[cohort_idx]

        scaled = X_test_scaled[
            test_order
        ]

        (
            heatmap,
            pred_idx,
            probs,
            conv_name,
            native_length,
        ) = gradcam_1d(
            model,
            scaled
        )

        heatmaps[
            test_order
        ] = heatmap

        mask = (
            heatmap > args.threshold
        )

        masks[
            test_order
        ] = mask.astype(
            np.uint8
        )

        true_idx = int(
            y[cohort_idx]
        )

        pred_class = (
            CLASS_NAMES[pred_idx]
        )

        true_class = (
            CLASS_NAMES[true_idx]
        )

        confidence = float(
            probs[pred_idx]
        )

        row = {
            "test_order": test_order,
            "cohort_index": cohort_idx,
            "patient_id": patient_ids[
                cohort_idx
            ],
            "true_id": true_idx,
            "true_class": true_class,
            "pred_id": pred_idx,
            "pred_class": pred_class,
            "correct": (
                true_idx == pred_idx
            ),
            "confidence": confidence,
            "last_conv_layer": conv_name,
            "native_cam_length": (
                native_length
            ),
            "resized_cam_length": (
                len(heatmap)
            ),
            "threshold": (
                args.threshold
            ),
            "salient_samples": int(
                mask.sum()
            ),
            "salient_fraction": float(
                mask.mean()
            ),
            "heatmap_max": float(
                heatmap.max()
            ),
            "heatmap_mean": float(
                heatmap.mean()
            ),
        }

        for i, cls in enumerate(
            CLASS_NAMES
        ):
            row[
                f"prob_{cls}"
            ] = float(
                probs[i]
            )

        rows.append(row)

        # Save visualization candidates:
        # first correct + first error from each true class
        plot_candidates.append({
            **row,
            "raw": raw,
            "scaled": scaled,
            "heatmap": heatmap,
        })

        if (
            (test_order + 1) % 100 == 0
            or test_order == 0
        ):
            print(
                f"Processed "
                f"{test_order + 1}/"
                f"{len(test)}"
            )

    manifest = pd.DataFrame(
        rows
    )

    # ---------------------------------------------------------
    # Save arrays
    # ---------------------------------------------------------

    np.save(
        out / "heatmaps_1d.npy",
        heatmaps
    )

    np.save(
        out / "salient_masks_threshold_023.npy",
        masks
    )

    manifest.to_csv(
        out / "gradcam_1d_manifest.csv",
        index=False
    )

    # ---------------------------------------------------------
    # Save selected visual examples
    # first correct and first error for each class
    # ---------------------------------------------------------

    selected = []

    for cls in CLASS_NAMES:

        correct = [
            x for x in plot_candidates
            if x["true_class"] == cls
            and x["correct"]
        ]

        wrong = [
            x for x in plot_candidates
            if x["true_class"] == cls
            and not x["correct"]
        ]

        if correct:
            selected.append(
                (
                    f"{cls}_correct",
                    correct[0]
                )
            )

        if wrong:
            selected.append(
                (
                    f"{cls}_error",
                    wrong[0]
                )
            )

    for tag, item in selected:

        filename = (
            f"{tag}_"
            f"{item['patient_id']}.png"
        )

        save_plot(
            raw_signal=item["raw"],
            scaled_signal=item["scaled"],
            heatmap=item["heatmap"],
            patient_id=item[
                "patient_id"
            ],
            true_class=item[
                "true_class"
            ],
            pred_class=item[
                "pred_class"
            ],
            confidence=item[
                "confidence"
            ],
            output_path=(
                plots_dir / filename
            ),
            threshold=args.threshold,
        )

    # ---------------------------------------------------------
    # Audit
    # ---------------------------------------------------------

    zero_maps = int(
        np.sum(
            np.max(
                heatmaps,
                axis=1
            ) == 0
        )
    )

    finite = bool(
        np.isfinite(
            heatmaps
        ).all()
    )

    print("\n" + "=" * 78)
    print("GRAD-CAM AUDIT")
    print("=" * 78)

    print(
        "Test patients      :",
        len(test)
    )

    print(
        "Heatmap shape      :",
        heatmaps.shape
    )

    print(
        "Threshold          :",
        args.threshold
    )

    print(
        "Finite values      :",
        finite
    )

    print(
        "Zero heatmaps      :",
        zero_maps
    )

    print(
        "Last Conv1D        :",
        last_conv.name
    )

    print(
        "Native CAM length  :",
        rows[0][
            "native_cam_length"
        ]
    )

    print(
        "Correct predictions:",
        int(
            manifest[
                "correct"
            ].sum()
        )
    )

    print(
        "Accuracy from manifest:",
        float(
            manifest[
                "correct"
            ].mean()
        )
    )

    print(
        "\nMean salient fraction:"
    )

    print(
        manifest.groupby(
            "true_class"
        )[
            "salient_fraction"
        ].mean()
    )

    summary = {
        "n_test": int(len(test)),
        "heatmap_shape": [
            int(x)
            for x in heatmaps.shape
        ],
        "threshold": float(
            args.threshold
        ),
        "last_conv_layer": (
            last_conv.name
        ),
        "native_cam_length": int(
            rows[0][
                "native_cam_length"
            ]
        ),
        "zero_heatmaps": zero_maps,
        "finite": finite,
        "classification_accuracy": float(
            manifest[
                "correct"
            ].mean()
        ),
    }

    with open(
        out / "gradcam_summary.json",
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    if not finite:
        raise RuntimeError(
            "Non-finite Grad-CAM values detected"
        )

    print("\nOutputs:")
    print(
        out /
        "heatmaps_1d.npy"
    )
    print(
        out /
        "salient_masks_threshold_023.npy"
    )
    print(
        out /
        "gradcam_1d_manifest.csv"
    )
    print(
        plots_dir
    )

    print("\nSTATUS: PASS")


if __name__ == "__main__":
    main()
