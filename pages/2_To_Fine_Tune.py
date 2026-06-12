import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU mode
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import AllChem, SDWriter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pickle
import io
import base64

import joblib
import subprocess
import tempfile
import os

# === Load tokenizer safely ===
try:
    with open("smiles_tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)
except UnicodeDecodeError:
    with open("smiles_tokenizer.pkl", "rb") as f:
        tok = pickle.load(f, encoding="latin1")

char2idx = tok["char2idx"]
idx2char = tok["idx2char"]

vocab_size = len(char2idx)
max_len = 150
embedding_dim = 256
lstm_units = 512

# === Model architecture ===
def build_best_lstm_model(vocab_size, embedding_dim=256, lstm_units=512, max_len=150):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
        LSTM(lstm_units, return_sequences=True),
        Dropout(0.3),
        LSTM(lstm_units, return_sequences=True),
        Dropout(0.3),
        Dense(vocab_size, activation='softmax')
    ])
    return model
    
# === Validity ===
def is_valid(smiles):
    return Chem.MolFromSmiles(smiles) is not None
    
def augment_smiles(smiles, num_augmentations=5):
    """Generate randomized SMILES for one molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    return [Chem.MolToSmiles(mol, doRandom=True, canonical=False) for _ in range(num_augmentations)]
    
def augment_smiles_dataset(smiles_list, num_augmentations=5):
    """Augment a list of SMILES strings."""
    augmented = []
    for smi in smiles_list:
        augmented.append(smi)  # Include original
        augmented.extend(augment_smiles(smi, num_augmentations))
    return augmented
    
    
# === Sidebar Controls ===
st.sidebar.subheader("Data Augmentation")

use_augmentation = st.sidebar.checkbox("Enable SMILES Augmentation", value=False)

num_augmentations = None

if use_augmentation:
    num_augmentations = st.sidebar.number_input(
        "Enter number of augmentations per SMILES",
        min_value=1,
        max_value=20,
        value=3,
        step=1,
        placeholder="Type a number..."
    )

#fine tune settings
st.sidebar.header("Fine-tuning Settings")

epochs = st.sidebar.number_input(
    "Epochs",
    min_value=1,
    max_value=5000,
    value=5,
    step=1
)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128, 256, 512], index=1)
learning_rate = st.sidebar.selectbox("Learning Rate", [1e-3, 1e-4, 1e-5, 5e-4, 5e-5], index=3)

user_smiles = None    
uploaded_file = st.file_uploader("Upload CSV with SMILES", type=["csv"])

if uploaded_file:
    df_user = pd.read_csv(uploaded_file, encoding="latin1")

    if "SMILES" not in df_user.columns:
        st.error("CSV must contain 'SMILES' column")
    else:
        smiles_raw = df_user["SMILES"].dropna().tolist()

        # === Validity ===
        valid_smiles = [s for s in smiles_raw if is_valid(s)]
        invalid_count = len(smiles_raw) - len(valid_smiles)

        # === Uniqueness ===
        unique_smiles = list(set(valid_smiles))
        duplicates = len(valid_smiles) - len(unique_smiles)

        # === DO NOT OVER-FILTER (IMPORTANT FIX) ===
        allowed_chars = set(char2idx.keys())

        filtered_smiles = []
        skipped = 0

        for s in unique_smiles:
            if all(c in char2idx for c in s):
                filtered_smiles.append(s)
            else:
                skipped += 1

        # === Stats ===
        st.info(f"Total uploaded: {len(smiles_raw)}")
        st.info(f"Valid SMILES: {len(valid_smiles)}")
        st.info(f"Invalid removed: {invalid_count}")
        st.info(f"Duplicates removed: {duplicates}")
        st.warning(f"Skipped due to unknown tokens: {skipped}")
        st.success(f"Final SMILES used for training: {len(filtered_smiles)}")

        # === Apply augmentation ===
        if use_augmentation:
            st.info(f"Applying augmentation ({num_augmentations} per SMILES)...")

            augmented_smiles = augment_smiles_dataset(
                filtered_smiles,
                num_augmentations=num_augmentations
            )

            st.success(f"Total after augmentation: {len(augmented_smiles)}")

            user_smiles = augmented_smiles
        else:
            user_smiles = filtered_smiles

        # === Remove duplicates again ===
        user_smiles = list(set(user_smiles))

        # === Shuffle (important for training) ===
        import random
        random.shuffle(user_smiles)
        st.info(f"Final number of smiles used for Fine tunning: {len(user_smiles)}")
        
              
def tokenize_smiles(smiles_list, char2idx, max_len):
    sequences = []
    for smi in smiles_list:
        seq = [char2idx.get(c, 0) for c in smi]
        seq = seq[:max_len]
        seq += [0] * (max_len - len(seq))
        sequences.append(seq)
    return np.array(sequences)

   
model = build_best_lstm_model(vocab_size, max_len)
model.build(input_shape=(None, max_len))
model.load_weights("Final_base_model.weights.h5")

#model.compile(
#    loss="sparse_categorical_crossentropy",
#    optimizer=Adam(learning_rate=0.0005)
#)

import streamlit as st
from tensorflow.keras.callbacks import Callback

class StreamlitProgressCallback(Callback):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        loss = logs.get("loss", 0)
        val_loss = logs.get("val_loss", 0)

        acc = logs.get("accuracy", 0)
        val_acc = logs.get("val_accuracy", 0)

        # Progress bar
        progress = (epoch + 1) / self.epochs
        self.progress_bar.progress(progress)

        # Text output
        self.status_text.text(
            f"Epoch {epoch+1}/{self.epochs} | "
            f"Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Acc: {acc:.4f} | Val Acc: {val_acc:.4f}"
        )
        

checkpoint_f = ModelCheckpoint(
    "Fine_tune_model.weights.h5",
    save_best_only=True,
    save_weights_only=True,
    monitor="val_loss",
    mode="min",
    verbose=1
)

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def fine_tune_model(model, smiles_list, char2idx, max_len, epochs, batch_size, lr):

    # === Tokenization ===
    tokens = [[char2idx[c] for c in s] for s in smiles_list]

    X = pad_sequences(tokens, maxlen=max_len, padding='post')
    y = np.zeros_like(X)
    y[:, :-1] = X[:, 1:]
    y[:, -1] = 0

    # === Train/Val Split ===
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # === Compile with user LR ===
    optimizer = Adam(learning_rate=lr)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    # === Callbacks ===
    
    progress_cb = StreamlitProgressCallback(epochs)
    earlystop = EarlyStopping(patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    # === Training ===
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[checkpoint_f, progress_cb, earlystop, reduce_lr]
    )

    return model
    
    
if user_smiles and st.button("Fine-tune Model"):

    with st.spinner("Fine-tuning model..."):

        model = fine_tune_model(
            model,
            user_smiles,
            char2idx,
            max_len,
            epochs=epochs,
            batch_size=batch_size,
            lr=learning_rate
        )

    st.success("Model fine-tuned successfully!")
    
    model.load_weights("Fine_tune_model.weights.h5")

    # SAVE TO MEMORY (NOT DISK)
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        model.save(tmp.name)
        with open(tmp.name, "rb") as f:
            st.session_state["model_bytes"] = f.read()
            
if "model_bytes" in st.session_state:

    st.download_button(
        "Download Fine-tuned Model",
        st.session_state["model_bytes"],
        "Fine_tune_model.weights.h5"
    )