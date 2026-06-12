# Molecule Generator

A deep learning-based molecular generation platform for the de novo design of small molecules using Long Short-Term Memory (LSTM) networks and SMILES representations.

The framework supports:

* Training a baseline molecular generative model from user-provided datasets.
* Fine-tuning the pretrained model on target-specific datasets.
* Generating novel molecules using temperature-controlled sampling.
* Interactive molecule generation through a Streamlit web interface.

---

## Features

* Character-level LSTM molecular generator
* User-defined training datasets
* Fine-tuning on target-specific molecules
* Adjustable sampling temperature
* Generation of novel SMILES strings
* Export generated molecules to CSV
* Streamlit-based graphical user interface

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Soniakkn/Molecule_Generator.git
cd Molecule_Generator
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Workflow

### Step 1: Prepare Dataset

Input CSV must contain a column named:

```text
smiles
```

Run preprocessing:

```bash
python prepare_smiles.py \
    --input dataset.csv
```

Outputs:

```text
filtered_smiles.csv
tokenizer.pkl
```

The preprocessing step:

* Removes duplicate molecules
* Removes salts
* Filters molecules by molecular weight
* Builds a tokenizer for SMILES encoding

---

### Step 2: Train Baseline Model

Train the LSTM generator:

```bash
python train_base_model.py \
    --smiles filtered_smiles.csv \
    --tokenizer tokenizer.pkl
```

Optional parameters:

```bash
--epochs
--batch_size
--learning_rate
--embedding_dim
--lstm_units
--dropout
--output
```

Example:

```bash
python train_base_model.py \
    --smiles filtered_smiles.csv \
    --tokenizer tokenizer.pkl \
    --epochs 200 \
    --learning_rate 0.001
```

Output:

```text
Final_base_model.weights.h5
```

---

### Step 3: Fine-Tune the Model

Fine-tune on a target-specific dataset:

```bash
python fine_tune_model.py \
    --smiles target_dataset.csv \
    --tokenizer tokenizer.pkl
```

By default the script uses:

```text
Final_base_model.weights.h5
```

To use a custom pretrained model:

```bash
python fine_tune_model.py \
    --smiles target_dataset.csv \
    --tokenizer tokenizer.pkl \
    --base_weights my_model.weights.h5
```

Output:

```text
fine_tuned_model.weights.h5
```

---

### Step 4: Generate Molecules

Generate molecules using the trained model:

```bash
python generate_molecules.py \
    --tokenizer tokenizer.pkl \
    --weights fine_tuned_model.weights.h5 \
    --num_samples 1000 \
    --temperature 1.0
```

Output:

```text
generated_smiles.csv
```

---

## Streamlit Web Application

A graphical user interface is available through Streamlit.

Launch locally:

```bash
streamlit run app.py
```

The application allows users to:

* Generate molecules interactively
* Select baseline or fine-tuned models
* Adjust sampling temperature
* Download generated molecules as CSV files
* Fine-tune models on custom datasets

---


## Developed By

**Sonia Kumari**

### Under the Guidance of

**Dr. M. Elizabeth Sobhia**

Department of Pharmacoinformatics

National Institute of Pharmaceutical Education and Research (NIPER), Mohali

