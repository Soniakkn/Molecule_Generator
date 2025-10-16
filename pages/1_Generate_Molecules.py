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
import pickle
import io
import base64

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


# === Load model function with caching ===
@st.cache_resource
def load_selected_model(choice):
    model = build_best_lstm_model(vocab_size, max_len)
    model.build(input_shape=(None, max_len))

    if choice == "Fine-tuned on Colchicine Inhibitors":
        weights_path = "NEW_FT_WITH_MERGED_v2_augumented_20_filtered_data.weights.h5"
    else:
        weights_path = "GPU__best_model_MERGED_v2_Again_run2_log_32_batch_0_0005_lr.weights.h5"

    model.load_weights(weights_path)
    return model


# === Sampling function ===
def sample_smiles(model, start="C", num_samples=10, temperature=1.0, max_len=150):
    generated = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(num_samples):
        seq = [char2idx.get(c, 0) for c in start]
        for _ in range(max_len - len(seq)):
            padded = pad_sequences([seq], maxlen=max_len, padding="post")
            preds = model.predict(padded, verbose=0)[0][len(seq)-1]

            preds = np.asarray(preds).astype("float64")
            preds = np.log(preds + 1e-8) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)

            next_idx = np.random.choice(len(preds), p=preds)
            if next_idx == 0:
                break
            seq.append(next_idx)

        smiles = "".join([idx2char[i] for i in seq if i != 0])
        generated.append(smiles)

        # --- Update progress bar ---
        progress = int((i + 1) / num_samples * 100)
        progress_bar.progress(progress / 100)
        status_text.text(f"Generating molecules... {i + 1}/{num_samples} complete")

    progress_bar.empty()
    status_text.text("? Generation complete!")
    return generated



# === Validate SMILES ===
def validate_smiles(smiles_list):
    valid, invalid = [], []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            valid.append(smi)
        else:
            invalid.append(smi)
    return valid, invalid


# === Compute SA score ===
try:
    import sascorer

    def calculate_sa_score(mol):
        try:
            score = sascorer.calculateScore(mol)
            return round(float(score), 3)
        except Exception:
            return None

except Exception as e:
    st.warning(f"SA_Score module not found: {e}")
    def calculate_sa_score(mol):
        return None

# === Convert valid SMILES to SDF ===
def smiles_to_sdf(valid_smiles):
    mols = [Chem.MolFromSmiles(s) for s in valid_smiles if Chem.MolFromSmiles(s)]
    for m in mols:
        AllChem.EmbedMolecule(m, AllChem.ETKDG())
    sdf_buffer = io.StringIO()
    writer = SDWriter(sdf_buffer)
    for m in mols:
        writer.write(m)
    writer.close()
    return sdf_buffer.getvalue().encode("utf-8")

# === Helper: Auto-download link ===
def auto_download_link(data, filename, mime_type):
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}">?? Download {filename}</a>'
    st.markdown(href, unsafe_allow_html=True)

# ===== Streamlit UI =====
st.title("Molecule Generator")
st.write("Generate molecules using your trained LSTM models.")

# --- Sidebar ---
st.sidebar.header("Model Settings")
model_choice = st.sidebar.radio(
    "Select Model:",
    ("Base Model", "Fine-tuned on Colchicine Inhibitors")
)
temperature = st.sidebar.slider(
    "Sampling Temperature",
    min_value=0.1,
    max_value=2.0,
    value=1.0,
    step=0.1,
    help="Lower = conservative molecules, Higher = more diverse molecules"
)
num = st.sidebar.number_input(
    "Number of molecules to generate",
    min_value=1,
    max_value=100000,
    value=10,
    step=1
)

# --- Load model ---
st.write(f"Loading **{model_choice}** ...")
model = load_selected_model(model_choice)
st.success("Model loaded successfully!")

# --- Generate molecules ---
# --- User Input for Reference Molecule ---
user_smile = st.text_input(
    "Enter a reference SMILES to compute Tanimoto similarity (optional):",
    value="",
    help="If provided, the similarity between this SMILES and generated molecules will be calculated."
)

# --- Generate molecules ---
if st.button("Generate Molecules"):
    # Reset session state for new generation
    st.session_state["df"] = pd.DataFrame()
    st.session_state["valid_mols"] = []
    st.session_state["generate_trigger"] = st.session_state.get("generate_trigger", 0) + 1

    st.info("Generating molecules... please wait.")
    smiles_list = sample_smiles(model, start="C", num_samples=num, temperature=temperature, max_len=max_len)
    valid, invalid = validate_smiles(smiles_list)

    # Compute SA scores
    sa_scores, valid_mols = [], []
    for smi in valid:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            valid_mols.append(mol)
            sa_scores.append(calculate_sa_score(mol))
        else:
            sa_scores.append(None)

    # --- Compute Tanimoto similarity only if user provided a valid SMILES ---
    df_data = {"SMILES": valid, "SA_Score": sa_scores}

    if user_smile.strip():  # check if user provided something
        ref_mol = Chem.MolFromSmiles(user_smile)
        if ref_mol:
            from rdkit.Chem import DataStructs
            ref_fp = Chem.RDKFingerprint(ref_mol)
            tanimoto_scores = []
            for m in valid_mols:
                fp = Chem.RDKFingerprint(m)
                sim = DataStructs.FingerprintSimilarity(ref_fp, fp)
                tanimoto_scores.append(round(sim, 3))
            df_data["Tanimoto_Similarity"] = tanimoto_scores
        else:
            st.warning("Invalid reference SMILES. Skipping Tanimoto similarity calculation.")

    # Store in session_state
    st.session_state["df"] = pd.DataFrame(df_data)
    st.session_state["valid_mols"] = valid
    st.session_state["invalid"] = invalid

# --- Display results ---
# --- Display results ---
if "df" in st.session_state and not st.session_state["df"].empty:
    df = st.session_state["df"]
    valid = st.session_state["valid_mols"]
    invalid = st.session_state.get("invalid", [])

    st.subheader(
        "Valid Molecules with SA Scores"
        + (" & Tanimoto Similarity" if "Tanimoto_Similarity" in df.columns else "")
    )
    st.dataframe(df)

    if invalid:
        st.subheader("Invalid Molecules")
        st.write(invalid)

    # --- CSV and SDF download buttons ---
    csv = df.to_csv(index=False).encode("utf-8")
    csv_filename = (
        "generated_smiles_with_SA_Tanimoto.csv"
        if "Tanimoto_Similarity" in df.columns
        else "generated_smiles_with_SA.csv"
    )
    st.download_button(
        "Download CSV",
        csv,
        csv_filename,
        "text/csv",
        key=f"csv_download_{st.session_state['generate_trigger']}"
    )

    sdf_data = smiles_to_sdf(valid)
    sdf_filename = "generated_molecules.sdf"
    st.download_button(
        "Download SDF",
        sdf_data,
        sdf_filename,
        "chemical/x-mdl-sdfile",
        key=f"sdf_download_{st.session_state['generate_trigger']}"
    )

    # --- Automatic download only once ---
    if not st.session_state.get("downloaded", False):
        # Prepare CSV
        csv = df.to_csv(index=False).encode("utf-8")
        csv_b64 = base64.b64encode(csv).decode()

        # Prepare SDF
        sdf_data = smiles_to_sdf(valid)
        sdf_b64 = base64.b64encode(sdf_data).decode()

        # Inject HTML+JS to download both automatically
        html_code = f"""
        <a id="csv_link" href="data:text/csv;base64,{csv_b64}" download="generated_smiles.csv"></a>
        <a id="sdf_link" href="data:chemical/x-mdl-sdfile;base64,{sdf_b64}" download="generated_molecules.sdf"></a>
        <script>
        document.getElementById('csv_link').click();
        document.getElementById('sdf_link').click();
        </script>
        """
        st.components.v1.html(html_code, height=0, width=0)

        # Mark as downloaded
        st.session_state["downloaded"] = True