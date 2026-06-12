```python
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
    default=256
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

args = parser.parse_args()


# ====================================================
# Load tokenizer
# ====================================================

with open(args.tokenizer, "rb") as f:
    tok = pickle.load(f)

char2idx = tok["char2idx"]
idx2char = tok["idx2char"]

vocab_size = tok["vocab_size"]
max_len = tok["max_len"]


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

for _ in range(
    args.num_samples
):

    generated.append(
        sample_smiles(
            model
        )
    )

df = pd.DataFrame(
    generated,
    columns=["smiles"]
)

df.to_csv(
    args.output,
    index=False
)

print(
    f"\nGenerated "
    f"{len(df)} molecules"
)

print(
    f"Saved to "
    f"{args.output}"
)
```
