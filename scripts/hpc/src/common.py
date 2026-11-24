from __future__ import annotations
import os
import random
import json
from pathlib import Path
import numpy as np

CLASS_NAMES = ["AFib", "GSVT", "SB", "SR"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

ALIASES = {
    "afib": "AFib",
    "af": "AFib",
    "atrialfibrillation": "AFib",
    "gsvt": "GSVT",
    "svt": "GSVT",
    "sb": "SB",
    "sinusbradycardia": "SB",
    "sr": "SR",
    "sinusrhythm": "SR",
}

def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

def canonical_label(x):
    if isinstance(x, bytes):
        x = x.decode("utf-8")
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)) and float(x).is_integer():
        return int(x)
    s = str(x).strip()
    if s in CLASS_TO_ID:
        return CLASS_TO_ID[s]
    key = "".join(ch for ch in s.lower() if ch.isalnum())
    if key in ALIASES:
        return CLASS_TO_ID[ALIASES[key]]
    try:
        v = int(float(s))
        return v
    except Exception:
        raise ValueError(f"Unknown label: {x!r}")

def decode_labels(y):
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 4:
        return np.argmax(y, axis=1).astype(int)
    y = y.reshape(-1)
    return np.array([canonical_label(v) for v in y], dtype=int)

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)

def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
