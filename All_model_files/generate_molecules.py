
#!/usr/bin/env python3

import argparse
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from rdkit import Chem


def is_valid_smiles(smiles):

    try:

        mol = Chem.MolFromSmiles(smiles)

        return mol is not None

    except:

        return False
        
def canonicalize_smiles(smiles):

    try:

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return None

        return Chem.MolToSmiles(
            mol,
            canonical=True
        )

    except:

        return None


# ====================================================
# Arguments
# ====================================================

parser = argparse.ArgumentParser(
    description="Generate molecules from trained LSTM model"
)

parser.add_argument(
    "--tokenizer",
    required=True,
    help="Tokenizer file"
)

parser.add_argument(
    "--weights",
    default="Final_CBSIs_fine_tuned_model.weights.h5",
    help="Model weights"
)

parser.add_argument(
    "--num_samples",
    type=int,
    default=100,
    help="Number of molecules to generate"
)

parser.add_argument(
    "--temperature",
    type=float,
    default=1.0,
    help="Sampling temperature"
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
    default="generated_smiles.csv"
)

parser.add_argument(
    "--device",
    choices=["auto", "cpu", "gpu"],
    default="auto",
    help="Device to use: auto, cpu, or gpu"
)

args = parser.parse_args()

# ====================================================
# Device Selection
# ====================================================

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
        f"\nUsing GPU ({len(gpus)} detected)"
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
        
# ====================================================
# Load tokenizer
# ====================================================

with open(args.tokenizer, "rb") as f:
    tok = pickle.load(f)

char2idx = tok["char2idx"]
idx2char = tok["idx2char"]

#vocab_size = tok["vocab_size"]
#max_len = tok["max_len"]

vocab_size = len(char2idx)

# Set manually or calculate from data
max_len = 150

# ====================================================
# Build model
# ====================================================

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


with tf.device(device_name):

    model = build_model()

    model.build(
        input_shape=(None, max_len)
    )

    model.load_weights(
        args.weights
    )

print(
    f"Loaded model weights:\n"
    f"{args.weights}"
)


# ====================================================
# Generation
# ====================================================

def sample_smiles(
    model,
    start="C"
):

    seq = [
        char2idx.get(c, 0)
        for c in start
    ]

    for _ in range(
        max_len - len(seq)
    ):

        padded = pad_sequences(
            [seq],
            maxlen=max_len,
            padding="post"
        )

        with tf.device(device_name):

            preds = model.predict(
                padded,
                verbose=0
            )[0][len(seq)-1]

        preds = (
            np.log(preds + 1e-8)
            / args.temperature
        )

        preds = np.exp(preds)

        preds /= np.sum(preds)

        next_idx = np.random.choice(
            len(preds),
            p=preds
        )

        if next_idx == 0:
            break

        seq.append(next_idx)

    return "".join(
        idx2char[i]
        for i in seq
        if i != 0
    )

# ====================================================
# Generate molecules
# ====================================================

generated = []

for _ in range(args.num_samples):

    generated.append(
        sample_smiles(model)
    )

total_generated = len(generated)

# ---------------------------------------------
# Validate + Canonicalize
# ---------------------------------------------

valid_smiles = []

for smi in generated:

    canon = canonicalize_smiles(smi)

    if canon is not None:

        valid_smiles.append(canon)

total_valid = len(valid_smiles)

# ---------------------------------------------
# Unique Canonical Molecules
# ---------------------------------------------

unique_smiles = list(
    set(valid_smiles)
)

total_unique = len(unique_smiles)

# ---------------------------------------------
# Save
# ---------------------------------------------

df = pd.DataFrame(
    unique_smiles,
    columns=["smiles"]
)

df.to_csv(
    args.output,
    index=False
)

# ---------------------------------------------
# Statistics
# ---------------------------------------------

validity = (
    100 * total_valid / total_generated
)

uniqueness = (
    100 * total_unique / total_valid
    if total_valid > 0 else 0
)

print("\nGeneration Statistics")
print("-" * 40)

print(
    f"Total generated : {total_generated}"
)

print(
    f"Valid molecules : {total_valid}"
)

print(
    f"Unique molecules: {total_unique}"
)

print(
    f"Validity (%)    : {validity:.2f}"
)

print(
    f"Uniqueness (%)  : {uniqueness:.2f}"
)

print(
    f"\nSaved {total_unique} unique canonical SMILES"
)

print(
    f"Output file: {args.output}"
)

