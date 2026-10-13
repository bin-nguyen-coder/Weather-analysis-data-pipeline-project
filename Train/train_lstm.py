"""LSTM temperature forecaster - training loop.

Consumes the windowed dataset produced by
``Train/feature_engineering.py`` (``data/windows.npz``) and trains a
stacked Keras LSTM to predict the next-hour normalized temperature.

Artifacts written
-----------------
* ``Model/lstm_weather_model.h5`` : best checkpoint (lowest validation MSE)
  via ``ModelCheckpoint``.
* ``Model/scaler.pkl``            : the feature/target ``StandardScaler``
  produced by ``Transform/transform_features.py`` (validated and re-persisted
  here so the ``Model/`` folder is a self-contained deployment unit).
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Workflow.config import load_config  # noqa: E402

logger = logging.getLogger(__name__)


def load_windows(windows_path: str | Path) -> Dict[str, Any]:
    """Load ``data/windows.npz`` produced by feature engineering."""
    windows_path = Path(windows_path)
    if not windows_path.exists():
        raise FileNotFoundError(
            f"{windows_path} not found - run Train/feature_engineering.py first."
        )
    with np.load(windows_path) as data:
        return {
            "X": data["X"],
            "y": data["y"],
            "cities": data["cities"],
            "timestamps": data["timestamps"],
            "feature_names": data["feature_names"].tolist(),
            "lookback": int(data["lookback"][()]),
        }


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    validation_split: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle with a fixed seed and split into train / validation sets."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    split_at = int(len(X) * (1.0 - validation_split))
    train_idx, valid_idx = indices[:split_at], indices[split_at:]
    return X[train_idx], y[train_idx], X[valid_idx], y[valid_idx]


def build_lstm_model(
    n_features: int,
    lookback: int = 168,
    lstm_units: Tuple[int, ...] = (64, 32),
    dropout: float = 0.2,
    learning_rate: float = 0.001,
):
    """Build the Keras Sequential LSTM model (input 3D windows)."""
    from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    model = Sequential(name="lstm_weather_forecaster")
    model.add(Input(shape=(lookback, n_features)))
    for index, units in enumerate(lstm_units):
        return_sequences = index < len(lstm_units) - 1
        model.add(LSTM(units, return_sequences=return_sequences, name=f"lstm_{index}"))
        model.add(Dropout(dropout, name=f"dropout_{index}"))
    model.add(Dense(1, name="temperature_output"))
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def ensure_scaler(scaler_path: str | Path) -> None:
    """Validate that ``Model/scaler.pkl`` exists and re-persist it.

    The scaler is fitted during ``Transform/transform_features.py``; this
    training step just guarantees the artifact is present alongside the
    model so ``Forecast/predict.py`` can invert predictions.
    """
    scaler_path = Path(scaler_path)
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"{scaler_path} not found - run Transform/transform_features.py "
            "before training (the target temperature must share the scaler "
            "used to normalize the features)."
        )
    with open(scaler_path, "rb") as handle:
        scaler = pickle.load(handle)
    # Round-trip: rewrite the very same object so the Model/ folder is the
    # canonical deployment unit.
    with open(scaler_path, "wb") as handle:
        pickle.dump(scaler, handle)
    logger.info("Validated & re-saved scaler at %s", scaler_path)


def run(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Train the LSTM on the windowed dataset and persist the artifacts."""
    config = config if config is not None else load_config()
    pipeline = config["pipeline"]
    windows_path = Path(config["paths"]["windows"])
    model_path = Path(config["paths"]["model"])
    scaler_path = Path(config["paths"]["scaler"])

    epochs = int(pipeline.get("train_epochs", 30))
    batch_size = int(pipeline.get("train_batch_size", 64))
    validation_split = float(pipeline.get("validation_split", 0.2))
    seed = int(pipeline.get("random_seed", 42))

    data = load_windows(windows_path)
    X, y = data["X"], data["y"]
    lookback = data["lookback"]
    n_features = X.shape[2]
    model_path.parent.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_valid, y_valid = split_dataset(
        X, y, validation_split=validation_split, seed=seed
    )
    logger.info(
        "Training samples=%d validation_samples=%d (lookback=%d, features=%d)",
        len(X_train), len(X_valid), lookback, n_features,
    )

    import tensorflow as tf
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
    )

    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = build_lstm_model(n_features=n_features, lookback=lookback)

    callbacks = [
        ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1
        ),
    ]

    logger.info("Starting LSTM training (epochs=%d, batch_size=%d)...", epochs, batch_size)
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Re-load the best checkpoint (final weights == best validation weights).
    from tensorflow.keras.models import load_model

    # Thêm compile=False để tránh lỗi không tương thích optimizer khi load lại model HDF5 cũ
    best_model = load_model(str(model_path), compile=False)
    
    val_loss = float(history.history["val_loss"][-1])
    val_mae = float(history.history.get("val_mae", [float("nan")])[-1])
    logger.info("Training finished. val_loss=%.4f val_mae=%.4f", val_loss, val_mae)

    ensure_scaler(scaler_path)

    return {
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "train_samples": int(len(X_train)),
        "validation_samples": int(len(X_valid)),
        "val_loss": val_loss,
        "val_mae": val_mae,
        "epochs_trained": len(history.history["loss"]),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    report = run()
    print(report)
