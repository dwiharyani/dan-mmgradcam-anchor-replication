from pathlib import Path
import argparse
import json
import hashlib

import h5py
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    GlobalAveragePooling1D,
    Dropout,
    Dense,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


CLASS_NAMES = ["AFib", "GSVT", "SB", "SR"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_model():
    inputs = Input(
        shape=(5000, 1),
        name="ecg_input"
    )

    x = Conv1D(
        64, 5,
        activation="relu",
        padding="same",
        name="conv1d_1"
    )(inputs)

    x = MaxPooling1D(
        pool_size=2,
        name="pool1"
    )(x)

    x = Conv1D(
        128, 3,
        activation="relu",
        padding="same",
        name="conv1d_2"
    )(x)

    x = MaxPooling1D(
        pool_size=2,
        name="pool2"
    )(x)

    x = Conv1D(
        256, 5,
        activation="relu",
        padding="same",
        name="conv1d_3"
    )(x)

    x = MaxPooling1D(
        pool_size=2,
        name="pool3"
    )(x)

    x = Conv1D(
        256, 3,
        activation="relu",
        padding="same",
        name="conv1d_4"
    )(x)

    x = MaxPooling1D(
        pool_size=2,
        name="pool4"
    )(x)

    x = Conv1D(
        128, 5,
        activation="relu",
        padding="same",
        name="conv1d_5"
    )(x)

    x = MaxPooling1D(
        pool_size=2,
        name="pool5"
    )(x)

    x = Conv1D(
        128, 3,
        activation="relu",
        padding="same",
        name="conv1d_6"
    )(x)

    x = GlobalAveragePooling1D(
        name="global_average_pooling"
    )(x)

    x = Dropout(
        0.2,
        name="dropout_1"
    )(x)

    x = Dense(
        64,
        activation="relu",
        name="dense_1"
    )(x)

    x = Dropout(
        0.3,
        name="dropout_2"
    )(x)

    x = Dense(
        64,
        activation="relu",
        name="dense_2"
    )(x)

    outputs = Dense(
        4,
        activation="softmax",
        name="classifier"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="MM_GradCAM_1D_AUTHOR_CODE"
    )

    model.compile(
        optimizer=Adam(
            learning_rate=0.0003
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--h5",
        required=True
    )

    parser.add_argument(
        "--split",
        default="splits/frozen_split_author_code.csv"
    )

    parser.add_argument(
        "--output",
        default="results/1d_author_code"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Reproducibility
    # Author notebook did not explicitly fix TF seed.
    # We fix one for our reproducible HPC run.
    # ---------------------------------------------------------
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(
        args.seed
    )

    gpus = tf.config.list_physical_devices(
        "GPU"
    )

    print("=" * 78)
    print("MM-GRADCAM 1D — AUTHOR_CODE REPRODUCTION")
    print("=" * 78)

    print("TensorFlow :", tf.__version__)
    print("GPUs       :", gpus)
    print("Seed       :", args.seed)

    # ---------------------------------------------------------
    # LOAD FINAL LOCKED DATA
    # ---------------------------------------------------------
    with h5py.File(args.h5, "r") as f:
        X = f["X"][:]
        y = f["y"][:]
        patient_ids = f["patient_id"][:]

    patient_ids = np.array([
        x.decode() if isinstance(x, bytes) else str(x)
        for x in patient_ids
    ])

    print("\nHDF5:")
    print(args.h5)

    print("X:", X.shape)
    print("y:", y.shape)

    assert X.shape == (10588, 5000)
    assert y.shape == (10588,)

    print(
        "HDF5 SHA256:",
        sha256_file(args.h5)
    )

    # ---------------------------------------------------------
    # LOAD FROZEN SPLIT
    # ---------------------------------------------------------
    split = pd.read_csv(args.split)

    assert len(split) == 10588
    assert split["cohort_index"].nunique() == 10588

    train = (
        split[split["split"] == "train"]
        .sort_values("split_order")
        .copy()
    )

    val = (
        split[split["split"] == "val"]
        .sort_values("split_order")
        .copy()
    )

    test = (
        split[split["split"] == "test"]
        .sort_values("split_order")
        .copy()
    )

    assert len(train) == 8475
    assert len(val) == 1059
    assert len(test) == 1054

    train_idx = train["cohort_index"].to_numpy(dtype=int)
    val_idx = val["cohort_index"].to_numpy(dtype=int)
    test_idx = test["cohort_index"].to_numpy(dtype=int)

    # ---------------------------------------------------------
    # AUTHOR-CODE SCALING
    #
    # IMPORTANT:
    # Original notebook fits StandardScaler on ALL 9534
    # non-test samples BEFORE train/val split.
    #
    # We intentionally reproduce that behavior.
    # ---------------------------------------------------------
    pool_idx = np.concatenate([
        train_idx,
        val_idx
    ])

    scaler = StandardScaler()

    scaler.fit(
        X[pool_idx]
    )

    X_train = scaler.transform(
        X[train_idx]
    ).astype(np.float32)

    X_val = scaler.transform(
        X[val_idx]
    ).astype(np.float32)

    X_test = scaler.transform(
        X[test_idx]
    ).astype(np.float32)

    y_train_raw = y[train_idx]
    y_val_raw = y[val_idx]
    y_test_raw = y[test_idx]

    # Conv1D channel dimension
    X_train = X_train[..., None]
    X_val = X_val[..., None]
    X_test = X_test[..., None]

    y_train = to_categorical(
        y_train_raw,
        num_classes=4
    )

    y_val = to_categorical(
        y_val_raw,
        num_classes=4
    )

    y_test = to_categorical(
        y_test_raw,
        num_classes=4
    )

    print("\nData shapes:")
    print("Train:", X_train.shape, y_train.shape)
    print("Val  :", X_val.shape, y_val.shape)
    print("Test :", X_test.shape, y_test.shape)

    print("\nTest counts:")
    for i, cls in enumerate(CLASS_NAMES):
        print(
            cls,
            int(np.sum(y_test_raw == i))
        )

    # Save scaler statistics
    np.savez(
        out / "standard_scaler.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
        var=scaler.var_,
    )

    # ---------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------
    model = build_model()

    model.summary()

    with open(
        out / "model_summary.txt",
        "w"
    ) as fh:
        model.summary(
            print_fn=lambda s: fh.write(
                s + "\n"
            )
        )

    print("\nTrainable parameters:")
    print(model.count_params())

    # ---------------------------------------------------------
    # TRAIN
    #
    # No early stopping.
    # Exact notebook:
    # 100 epochs
    # batch size = 16
    # shuffle = default True
    # ---------------------------------------------------------
    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_val,
            y_val
        ),
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        verbose=2,
    )

    # AUTHOR notebook saves FINAL epoch model.
    model_path = (
        out /
        "trained_model.keras"
    )

    model.save(
        model_path,
        include_optimizer=True
    )

    # ---------------------------------------------------------
    # HISTORY
    # ---------------------------------------------------------
    history_df = pd.DataFrame(
        history.history
    )

    history_df.insert(
        0,
        "epoch",
        np.arange(
            1,
            len(history_df) + 1
        )
    )

    history_df.to_csv(
        out / "training_history.csv",
        index=False
    )

    # ---------------------------------------------------------
    # TEST
    # ---------------------------------------------------------
    test_loss, test_acc = model.evaluate(
        X_test,
        y_test,
        batch_size=args.batch_size,
        verbose=0,
    )

    probs = model.predict(
        X_test,
        batch_size=args.batch_size,
        verbose=0,
    )

    pred = np.argmax(
        probs,
        axis=1
    )

    accuracy = accuracy_score(
        y_test_raw,
        pred
    )

    macro_f1 = f1_score(
        y_test_raw,
        pred,
        average="macro"
    )

    uar = recall_score(
        y_test_raw,
        pred,
        average="macro"
    )

    weighted_f1 = f1_score(
        y_test_raw,
        pred,
        average="weighted"
    )

    cm = confusion_matrix(
        y_test_raw,
        pred,
        labels=[0, 1, 2, 3]
    )

    report = classification_report(
        y_test_raw,
        pred,
        labels=[0, 1, 2, 3],
        target_names=CLASS_NAMES,
        digits=6,
        output_dict=True,
        zero_division=0,
    )

    # ---------------------------------------------------------
    # PREDICTIONS
    # ---------------------------------------------------------
    pred_df = pd.DataFrame({
        "cohort_index": test_idx,
        "patient_id": patient_ids[test_idx],
        "true_id": y_test_raw,
        "true_class": [
            CLASS_NAMES[i]
            for i in y_test_raw
        ],
        "pred_id": pred,
        "pred_class": [
            CLASS_NAMES[i]
            for i in pred
        ],
        "correct": (
            pred == y_test_raw
        ),
    })

    for i, cls in enumerate(CLASS_NAMES):
        pred_df[
            f"prob_{cls}"
        ] = probs[:, i]

    pred_df.to_csv(
        out / "test_predictions.csv",
        index=False
    )

    pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    ).to_csv(
        out / "confusion_matrix.csv"
    )

    pd.DataFrame(
        report
    ).T.to_csv(
        out / "classification_report.csv"
    )

    metrics = {
        "protocol": "AUTHOR_CODE",
        "seed": args.seed,
        "tensorflow": tf.__version__,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": "Adam",
        "learning_rate": 0.0003,
        "loss": "categorical_crossentropy",
        "scaling": (
            "StandardScaler fit on "
            "train+validation pool (9534)"
        ),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "test_loss": float(test_loss),
        "test_accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "uar": float(uar),
        "h5_sha256": sha256_file(
            args.h5
        ),
    }

    with open(
        out / "metrics.json",
        "w"
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2
        )

    # ---------------------------------------------------------
    # FINAL OUTPUT
    # ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("TEST RESULTS — AUTHOR_CODE")
    print("=" * 78)

    print(
        f"Accuracy    : {accuracy:.6f}"
    )

    print(
        f"Macro-F1   : {macro_f1:.6f}"
    )

    print(
        f"Weighted-F1: {weighted_f1:.6f}"
    )

    print(
        f"UAR         : {uar:.6f}"
    )

    print(
        f"Test loss   : {test_loss:.6f}"
    )

    print("\nConfusion matrix:")
    print(cm)

    print("\nModel:")
    print(model_path.resolve())

    print("\nResults:")
    print(out.resolve())

    print("\nSTATUS: PASS")


if __name__ == "__main__":
    main()
