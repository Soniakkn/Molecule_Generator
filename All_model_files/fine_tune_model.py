
#!/usr/bin/env python3

import argparse
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split


# =========================================================
# Arguments
# =========================================================

parser = argparse.ArgumentParser(
    description="Fine-tune LSTM Molecular Generator"
)

parser.add_argument(
    "--smiles",
    required=True,
    help="Fine-tuning SMILES CSV"
)

parser.add_argument(
    "--tokenizer",
    required=True,
    help="Tokenizer file"
)

parser.add_argument(
    "--base_weights",
    default="base_model.weights.h5",
    help="Base model weights"
)

parser.add_argument(
    "--epochs",
    type=int,
    default=100
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=32
)

parser.add_argument(
    "--learning_rate",
    type=float,
    default=0.0001
)

parser.add_argument(
    "--embedding_dim",
    type=int,
    default=150
)

parser.add_argument(
    "--lstm_units",
    type=int,
    default=512
)

parser.add_argument(
    "--dropout",
    type=float,
    default=0.3
)

parser.add_argument(
    "--output",
    default="fine_tuned_model.weights.h5"
)

parser.add_argument(
    "--device",
    choices=["auto", "cpu", "gpu"],
    default="auto",
    help="Training device: auto, cpu, gpu"
)

args = parser.parse_args()

# =========================================================
# Device Selection
# =========================================================

if args.device == "cpu":

    tf.config.set_visible_devices([], "GPU")

    device_name = "/CPU:0"

    print("\nUsing CPU")

elif args.device == "gpu":

    gpus = tf.config.list_physical_devices("GPU")

    if len(gpus) == 0:
        raise RuntimeError(
            "GPU requested but no GPU detected."
        )

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(
            gpu,
            True
        )

    device_name = "/GPU:0"

    print(
        f"\nUsing GPU "
        f"({len(gpus)} detected)"
    )

else:

    gpus = tf.config.list_physical_devices("GPU")

    if len(gpus) > 0:

        for gpu in gpus:
            tf.config.experimental.set_memory_growth(
                gpu,
                True
            )

        device_name = "/GPU:0"

        print(
            f"\nAUTO mode: Using GPU "
            f"({len(gpus)} detected)"
        )

    else:

        device_name = "/CPU:0"

        print(
            "\nAUTO mode: GPU not found, using CPU"
        )

# =========================================================
# Load Data
# =========================================================

print(f"\nLoading fine-tuning dataset: {args.smiles}")

df = pd.read_csv(args.smiles)

if "smiles" not in df.columns:
    raise ValueError(
        "CSV must contain a column named 'smiles'"
    )

smiles_list = (
    df["smiles"]
    .astype(str)
    .str.strip()
    .tolist()
)

print(
    f"Fine-tuning molecules: "
    f"{len(smiles_list)}"
)


# =========================================================
# Load Tokenizer
# =========================================================

print(
    f"Loading tokenizer: "
    f"{args.tokenizer}"
)

with open(args.tokenizer, "rb") as f:
    tokenizer = pickle.load(f)

#char2idx = tokenizer["char2idx"]

#vocab_size = tokenizer["vocab_size"]

#max_len = tokenizer["max_len"]

char2idx = tokenizer["char2idx"]
idx2char = tokenizer["idx2char"]

vocab_size = len(char2idx)

# Set manually or calculate from data
max_len = 150

# =========================================================
# Tokenization
# =========================================================

def tokenize(smile):

    return [
        char2idx[c]
        for c in smile
    ]


tokenized = [
    tokenize(s)
    for s in smiles_list
]

input_seqs = pad_sequences(
    tokenized,
    maxlen=max_len,
    padding="post"
)

target_seqs = np.zeros_like(
    input_seqs
)

target_seqs[:, :-1] = input_seqs[:, 1:]

target_seqs[:, -1] = 0


# =========================================================
# Train / Validation Split
# =========================================================

X_train, X_val, y_train, y_val = train_test_split(
    input_seqs,
    target_seqs,
    test_size=0.25,
    random_state=42
)


# =========================================================
# Model Architecture
# =========================================================

def build_model():

    model = Sequential([

        Embedding(
            input_dim=vocab_size,
            output_dim=args.embedding_dim,
            mask_zero=True
        ),

        LSTM(
            args.lstm_units,
            return_sequences=True
        ),

        Dropout(
            args.dropout
        ),

        LSTM(
            args.lstm_units,
            return_sequences=True
        ),

        Dropout(
            args.dropout
        ),

        Dense(
            vocab_size,
            activation="softmax"
        )
    ])

    return model


# =========================================================
# Build Model
# =========================================================

with tf.device(device_name):

    model = build_model()

    model.build(
        input_shape=(None, max_len)
    )

with tf.device(device_name):

    print(
        f"\nLoading base model weights:\n"
        f"{args.base_weights}"
    )

    model.load_weights(
        args.base_weights
    )

optimizer = Adam(
    learning_rate=args.learning_rate,
    clipnorm=1.0
)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"]
)

model.summary()


# =========================================================
# Callbacks
# =========================================================

checkpoint_cb = ModelCheckpoint(
    filepath=args.output,
    save_best_only=True,
    save_weights_only=True,
    monitor="val_loss",
    mode="min",
    verbose=1
)

earlystop_cb = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr_cb = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    verbose=1
)


class BestEpochCallback(
    tf.keras.callbacks.Callback
):

    def __init__(self):

        self.best_epoch = None
        self.best_val_loss = float("inf")

    def on_epoch_end(
        self,
        epoch,
        logs=None
    ):

        val_loss = logs.get(
            "val_loss"
        )

        if (
            val_loss is not None
            and
            val_loss < self.best_val_loss
        ):

            self.best_val_loss = val_loss

            self.best_epoch = epoch + 1

            print(
                f"\nBest model so far "
                f"saved at epoch "
                f"{self.best_epoch}"
            )


best_epoch_cb = BestEpochCallback()


# =========================================================
# Fine-Tuning
# =========================================================

with tf.device(device_name):

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            checkpoint_cb,
            earlystop_cb,
            reduce_lr_cb,
            best_epoch_cb
        ],
        verbose=1
    )


print("\nFine-tuning completed.")

print(
    f"Best model saved as:\n"
    f"{args.output}"
)

