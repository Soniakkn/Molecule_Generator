import streamlit as st

st.title("About")

st.write("""
This project is a **Molecule Generator** based on deep learning.

- It uses an LSTM model trained on SMILES strings.
- Users can generate novel molecules interactively.
- Outputs are available in **CSV (SMILES)** and **SDF (3D structure)** formats.
- You can select either the base model or a fine-tuned model for colchicine inhibitors.
- The generation temperature can be adjusted for more conservative or diverse molecules.

This tool helps chemists and researchers explore novel chemical space quickly and efficiently.
""")