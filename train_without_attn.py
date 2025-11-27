import os
import json
import logging
import argparse
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import layers, models
from keras_nlp.layers import RETVecTokenizer
import pandas as pd


# ============================================================
# Setup run folder + logging
# ============================================================
def init_experiment(base_out_dir):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(base_out_dir, f"run_{stamp}")

    os.makedirs(f"{base_dir}/logs", exist_ok=True)
    os.makedirs(f"{base_dir}/reports", exist_ok=True)
    os.makedirs(f"{base_dir}/model", exist_ok=True)

    logging.basicConfig(
        filename=f"{base_dir}/logs/logs.txt",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logging.info("Experiment directory created.")
    return base_dir


# ============================================================
# Dataset Loader
# ============================================================
def load_dataset(path, batch_size):
    df = pd.read_csv(path)
    ds = tf.data.Dataset.from_tensor_slices(
        (df["text"].astype(str).tolist(), df["label"].astype(int).tolist())
    )
    return ds.batch(batch_size)


# ============================================================
# Build RETVec + BiLSTM Model
# ============================================================
def build_model(strategy):
    with strategy.scope():
        inputs = layers.Input(shape=(1,), dtype=tf.string)
        x = RETVecTokenizer(model="retvec-v1")(inputs)

        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        x = layers.Bidirectional(layers.LSTM(64))(x)

        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.2)(x)

        outputs = layers.Dense(1, activation="sigmoid")(x)

        model = models.Model(inputs, outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

    logging.info(model.summary(print_fn=lambda x: logging.info(x)))
    return model


# ============================================================
# Train
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="runs")

    args = parser.parse_args()

    base_dir = init_experiment(args.output_dir)

    strategy = tf.distribute.MirroredStrategy()

    train_ds = load_dataset(args.train_csv, args.batch_size)
    val_ds = load_dataset(args.val_csv, args.batch_size)

    model = build_model(strategy)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs
    )

    model.save(f"{base_dir}/model")

    with open(f"{base_dir}/reports/loss_curves.json", "w") as f:
        json.dump(
            {"train_loss": history.history["loss"],
             "val_loss": history.history["val_loss"]},
            f, indent=2
        )

    logging.info("Training completed.")


if __name__ == "__main__":
    main()
