import streamlit as st

st.title("About")

st.write("""
This project is a **Molecule Generator** based on deep learning for the de novo design of novel small molecules.

### Features
- Uses an LSTM-based generative model trained on SMILES strings.
- Generates novel molecules interactively.
- Supports both a baseline model and a fine-tuned model for colchicine-binding site inhibitors.
- Adjustable sampling temperature for controlling molecular diversity.
- Exports generated molecules in **CSV (SMILES)** and **SDF (3D structure)** formats.
- Enables rapid exploration of chemically relevant and drug-like molecular space.

### Developed By
**Sonia Kumari**

### Under the Guidance of
**Dr. M. Elizabeth Sobhia**

**Department of Pharmacoinformatics**  
**National Institute of Pharmaceutical Education and Research (NIPER), Mohali**

This platform was developed to facilitate AI-driven molecular design and accelerate the discovery of novel therapeutic candidates.
""")