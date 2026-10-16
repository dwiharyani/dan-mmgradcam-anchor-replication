from pathlib import Path
import argparse
import json
import re

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D,
    GlobalAveragePooling2D, Dense, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


CLASS_NAMES = ["AFib", "GSVT", "SB", "SR"]


def build_model():

    inputs = Input(shape=(300, 600, 1))

    x = Conv2D(64,  (5,5), activation="relu", padding="same")(inputs)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(256, (5,5), activation="relu", padding="same")(x)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(256, (3,3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(128, (5,5), activation="relu", padding="same")(x)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(128, (3,3), activation="relu", padding="same")(x)

    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)

    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)

    outputs = Dense(4, activation="softmax")(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=0.0003),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        default="data/2d_author_clean"
    )
    ap.add_argument(
        "--pairs",
        default="splits/mm_pairs.csv"
    )
    ap.add_argument(
        "--output",
        default="results/2d_author_code"
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=100
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    args = ap.parse_args()

    tf.keras.utils.set_random_seed(42)

    root = Path(args.root)
    train_dir = root / "Traindata"
    test_dir = root / "Testdata"

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # EXACT notebook preprocessing
    datagen = ImageDataGenerator(
        validation_split=0.111,
        rescale=1.0 / 255.0
    )

    train_data = datagen.flow_from_directory(
        train_dir,
        target_size=(300, 600),
        batch_size=args.batch_size,
        class_mode="categorical",
        color_mode="grayscale",
        subset="training",
        shuffle=True,
        seed=42,
    )

    valid_data = datagen.flow_from_directory(
        train_dir,
        target_size=(300, 600),
        batch_size=args.batch_size,
        class_mode="categorical",
        color_mode="grayscale",
        subset="validation",
        shuffle=True,
        seed=42,
    )

    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0
    )

    test_data = test_datagen.flow_from_directory(
        test_dir,
        target_size=(300, 600),
        batch_size=args.batch_size,
        class_mode="categorical",
        color_mode="grayscale",
        shuffle=False,
    )

    print("\nCLASS INDICES:")
    print(train_data.class_indices)

    assert train_data.class_indices == {
        "AFib": 0,
        "GSVT": 1,
        "SB": 2,
        "SR": 3,
    }

    print("\nSIZES:")
    print("Train:", train_data.samples)
    print("Val  :", valid_data.samples)
    print("Test :", test_data.samples)

    assert train_data.samples == 8476
    assert valid_data.samples == 1057
    assert test_data.samples == 1054

    model = build_model()

    print("\nParameters:", model.count_params())
    assert model.count_params() == 2464644

    checkpoint = ModelCheckpoint(
        out / "best_model.keras",
        monitor="val_accuracy",
        save_weights_only=False,
        save_best_only=True,
        mode="max",
        verbose=1,
    )

    history = model.fit(
        train_data,
        epochs=args.epochs,
        validation_data=valid_data,
        verbose=2,
        callbacks=[checkpoint],
    )

    model.save(
        out / "final_model.keras",
        include_optimizer=True
    )

    hist = pd.DataFrame(history.history)
    hist.insert(0, "epoch", np.arange(1, len(hist)+1))
    hist.to_csv(out / "training_history.csv", index=False)

    # evaluate BEST checkpoint
    best = tf.keras.models.load_model(
        out / "best_model.keras"
    )

    test_data.reset()

    probs = best.predict(
        test_data,
        verbose=1
    )

    pred = np.argmax(probs, axis=1)
    true = test_data.classes

    accuracy = accuracy_score(true, pred)
    macro_f1 = f1_score(true, pred, average="macro")
    weighted_f1 = f1_score(true, pred, average="weighted")
    uar = recall_score(true, pred, average="macro")

    cm = confusion_matrix(
        true,
        pred,
        labels=[0,1,2,3]
    )

    report = classification_report(
        true,
        pred,
        labels=[0,1,2,3],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    # Map generator filenames back to cohort IDs
    pairs = pd.read_csv(args.pairs).set_index("cohort_index")

    rows = []

    for i, filename in enumerate(test_data.filenames):

        m = re.search(r"hasta_(\d+)", filename)

        if not m:
            raise RuntimeError(
                f"Cannot parse cohort ID from {filename}"
            )

        cohort_index = int(m.group(1))

        r = pairs.loc[cohort_index]

        rows.append({
            "generator_order": i,
            "cohort_index": cohort_index,
            "patient_id": r["patient_id"],
            "true_class": CLASS_NAMES[int(true[i])],
            "pred_class": CLASS_NAMES[int(pred[i])],
            "correct": bool(pred[i] == true[i]),
            "image_filename": filename,
            **{
                f"prob_{CLASS_NAMES[j]}": float(probs[i,j])
                for j in range(4)
            }
        })

    pred_df = pd.DataFrame(rows)

    assert len(pred_df) == 1054
    assert pred_df["cohort_index"].nunique() == 1054

    pred_df.to_csv(
        out / "test_predictions.csv",
        index=False
    )

    pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    ).to_csv(
        out / "confusion_matrix.csv"
    )

    pd.DataFrame(report).T.to_csv(
        out / "classification_report.csv"
    )

    best_epoch = int(
        np.argmax(history.history["val_accuracy"]) + 1
    )

    metrics = {
        "protocol": "AUTHOR_DIRECTORY",
        "n_total_usable_images": 10587,
        "excluded_zero_byte": "hasta_6565.png",
        "n_train": 8476,
        "n_val": 1057,
        "n_test": 1054,
        "input_shape": [300, 600, 1],
        "rescale": "1/255",
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "optimizer": "Adam",
        "learning_rate": 0.0003,
        "best_epoch": best_epoch,
        "best_val_accuracy": float(
            max(history.history["val_accuracy"])
        ),
        "test_accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "uar": float(uar),
    }

    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "="*72)
    print("2D AUTHOR-DIRECTORY TEST RESULTS")
    print("="*72)

    print(f"Best epoch  : {best_epoch}")
    print(f"Accuracy    : {accuracy:.6f}")
    print(f"Macro-F1   : {macro_f1:.6f}")
    print(f"Weighted-F1: {weighted_f1:.6f}")
    print(f"UAR         : {uar:.6f}")

    print("\nConfusion matrix:")
    print(cm)

    print("\nSTATUS: PASS")


if __name__ == "__main__":
    main()
