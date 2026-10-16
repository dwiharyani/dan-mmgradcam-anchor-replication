from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dropout,
    Dense,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint


CLASS_NAMES = ["AFib", "GSVT", "SB", "SR"]
CLASS_TO_ID = {
    "AFib": 0,
    "GSVT": 1,
    "SB": 2,
    "SR": 3,
}


def load_image(path, label):
    img = tf.io.read_file(path)

    img = tf.io.decode_image(
        img,
        channels=1,
        expand_animations=False
    )

    img = tf.image.resize(
        img,
        [300, 600]
    )

    img = tf.cast(
        img,
        tf.float32
    ) / 255.0

    img.set_shape(
        [300, 600, 1]
    )

    return img, label


def make_dataset(df, batch_size, training=False):

    paths = df["image_path"].astype(str).values

    labels = np.array([
        CLASS_TO_ID[x]
        for x in df["class_name"]
    ], dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        ds = ds.shuffle(
            len(df),
            seed=42,
            reshuffle_each_iteration=True
        )

    ds = ds.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.batch(
        batch_size
    )

    ds = ds.prefetch(
        tf.data.AUTOTUNE
    )

    return ds


def build_model():

    inputs = Input(
        shape=(300, 600, 1),
        name="ecg_image"
    )

    x = Conv2D(
        64,
        (5, 5),
        activation="relu",
        padding="same",
        name="conv2d_1"
    )(inputs)

    x = MaxPooling2D(
        (2, 2),
        name="pool1"
    )(x)

    x = Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same",
        name="conv2d_2"
    )(x)

    x = MaxPooling2D(
        (2, 2),
        name="pool2"
    )(x)

    x = Conv2D(
        256,
        (5, 5),
        activation="relu",
        padding="same",
        name="conv2d_3"
    )(x)

    x = MaxPooling2D(
        (2, 2),
        name="pool3"
    )(x)

    x = Conv2D(
        256,
        (3, 3),
        activation="relu",
        padding="same",
        name="conv2d_4"
    )(x)

    x = MaxPooling2D(
        (2, 2),
        name="pool4"
    )(x)

    x = Conv2D(
        128,
        (5, 5),
        activation="relu",
        padding="same",
        name="conv2d_5"
    )(x)

    x = MaxPooling2D(
        (2, 2),
        name="pool5"
    )(x)

    x = Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same",
        name="conv2d_6"
    )(x)

    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.2)(x)

    x = Dense(
        64,
        activation="relu"
    )(x)

    x = Dropout(0.3)(x)

    x = Dense(
        64,
        activation="relu"
    )(x)

    outputs = Dense(
        4,
        activation="softmax"
    )(x)

    model = Model(
        inputs,
        outputs,
        name="MM_GradCAM_2D"
    )

    model.compile(
        optimizer=Adam(
            learning_rate=0.0003
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        default="splits/frozen_split_author_code.csv"
    )

    parser.add_argument(
        "--output",
        default="results/2d_author_code"
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

    tf.keras.utils.set_random_seed(42)

    out = Path(args.output)
    out.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(args.split)

    train = (
        df[df["split"] == "train"]
        .sort_values("split_order")
        .copy()
    )

    val = (
        df[df["split"] == "val"]
        .sort_values("split_order")
        .copy()
    )

    test = (
        df[df["split"] == "test"]
        .sort_values("split_order")
        .copy()
    )

    print("=" * 75)
    print("MM-GRADCAM 2D AUTHOR_CODE")
    print("=" * 75)

    print("Train :", len(train))
    print("Val   :", len(val))
    print("Test  :", len(test))

    assert len(train) == 8475
    assert len(val) == 1059
    assert len(test) == 1054

    train_ds = make_dataset(
        train,
        args.batch_size,
        training=True
    )

    val_ds = make_dataset(
        val,
        args.batch_size
    )

    test_ds = make_dataset(
        test,
        args.batch_size
    )

    model = build_model()

    model.summary()

    print(
        "\nParameters:",
        model.count_params()
    )

    checkpoint = ModelCheckpoint(
        out / "best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[checkpoint],
        verbose=2
    )

    model.save(
        out / "final_model.keras"
    )

    history_df = pd.DataFrame(
        history.history
    )

    history_df["epoch"] = np.arange(
        1,
        len(history_df) + 1
    )

    history_df.to_csv(
        out / "training_history.csv",
        index=False
    )

    # Evaluate best model
    best = tf.keras.models.load_model(
        out / "best_model.keras"
    )

    probs = best.predict(
        test_ds,
        verbose=1
    )

    pred = np.argmax(
        probs,
        axis=1
    )

    true = np.array([
        CLASS_TO_ID[x]
        for x in test["class_name"]
    ])

    accuracy = accuracy_score(
        true,
        pred
    )

    macro_f1 = f1_score(
        true,
        pred,
        average="macro"
    )

    weighted_f1 = f1_score(
        true,
        pred,
        average="weighted"
    )

    uar = recall_score(
        true,
        pred,
        average="macro"
    )

    cm = confusion_matrix(
        true,
        pred
    )

    report = classification_report(
        true,
        pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    pred_df = test[
        [
            "cohort_index",
            "patient_id",
            "class_name",
            "image_path"
        ]
    ].copy()

    pred_df["pred_id"] = pred

    pred_df["pred_class"] = [
        CLASS_NAMES[i]
        for i in pred
    ]

    pred_df["correct"] = (
        pred == true
    )

    for i, cls in enumerate(CLASS_NAMES):
        pred_df[f"prob_{cls}"] = probs[:, i]

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

    pd.DataFrame(
        report
    ).T.to_csv(
        out / "classification_report.csv"
    )

    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "uar": float(uar),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "input_shape": [300, 600, 1],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": 0.0003
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

    print("\n" + "=" * 75)
    print("2D TEST RESULTS")
    print("=" * 75)

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

    print("\nConfusion matrix:")
    print(cm)

    print("\nSTATUS: PASS")


if __name__ == "__main__":
    main()
