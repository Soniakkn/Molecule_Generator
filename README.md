# Molecule Generator

A deep learning-based molecular generation platform for the de novo design of small molecules using Long Short-Term Memory (LSTM) networks and SMILES representations.

The framework supports:

* Training a baseline molecular generative model from user-provided datasets.
* Fine-tuning the pretrained model on target-specific datasets.
* Generating novel molecules using temperature-controlled sampling.
* Evaluation of generated molecules using molecular descriptors and scaffold analysis.
* Interactive molecule generation through a Streamlit web interface.

---

## Features

* Character-level LSTM molecular generator
* User-defined training datasets
* Fine-tuning on target-specific molecules
* Adjustable sampling temperature
* CPU/GPU selection for training, fine-tuning, and generation
* Generation of novel SMILES strings
* Automatic validation, canonicalization, and duplicate removal
* Export generated molecules to CSV
* Molecular descriptor calculation and scaffold analysis
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
python All_model_files/prepare_smiles.py \
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
python All_model_files/train_base_model.py \
    --smiles filtered_smiles.csv \
    --tokenizer tokenizer.pkl \
    --device gpu
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
--device
```

Device options:

```text
auto    Automatically uses GPU if available, otherwise CPU
gpu     Force GPU execution
cpu     Force CPU execution
```

Example:

```bash
python All_model_files/train_base_model.py \
    --smiles filtered_smiles.csv \
    --tokenizer tokenizer.pkl \
    --epochs 200 \
    --learning_rate 0.001 \
    --device auto
```

Output:

```text
base_model.weights.h5
```

---

### Step 3: Fine-Tune the Model

Fine-tune on a target-specific dataset:

```bash
python All_model_files/fine_tune_model.py \
    --smiles target_dataset.csv \
    --tokenizer tokenizer.pkl \
    --device gpu
```

By default the script uses:

```text
base_model.weights.h5
```

To use a our pretrained model:

```bash
python All_model_files/fine_tune_model.py \
    --smiles target_dataset.csv \
    --tokenizer tokenizer.pkl \
    --base_weights Final_base_model.weights.h5 \
    --device gpu
```

Output:

```text
fine_tuned_model.weights.h5
```

---

### Step 4: Generate Molecules

Generate molecules using the trained model:

```bash
python All_model_files/generate_molecules.py \
    --tokenizer tokenizer.pkl \
    --weights fine_tuned_model.weights.h5 \
    --num_samples 1000 \
    --temperature 1.0 \
    --device auto
```

Output:

```text
generated_smiles.csv
```

The generation script automatically:

* Validates generated molecules
* Canonicalizes valid SMILES
* Removes duplicate molecules
* Reports:

  * Total generated molecules
  * Valid molecules
  * Unique molecules
  * Validity percentage
  * Uniqueness percentage

---

### Step 5: Evaluate Generated Molecules

Calculate molecular descriptors and scaffold information for generated molecules:

```bash
python All_model_files/evaluate.py \
    --input generated_smiles.csv \
    --output evaluated_molecules.csv
```

Output:

```text
evaluated_molecules.csv
```

The evaluation script calculates:

* Canonical SMILES
* Bemis-Murcko Scaffold
* Synthetic Accessibility (SA) Score
* Molecular Weight (MW)
* LogP
* Hydrogen Bond Acceptors (HBA)
* Hydrogen Bond Donors (HBD)
* Number of Chiral Centers
* Number of Rotatable Bonds

---

## Streamlit Web Application

A graphical user interface is available through Streamlit.

### Online Application

https://moleculegenerator-b25svymsww8uczfpur53yh.streamlit.app/

### Run Locally

```bash
streamlit run app.py
```

The application allows users to:

* Generate molecules interactively
* Upload custom pretrained models
* Select baseline or fine-tuned models
* Adjust sampling temperature
* Download generated molecules as CSV files
* Fine-tune models on custom datasets
* Evaluate generated molecules

---

## Developed By

**Sonia Kumari**

### Under the Guidance of

**Prof. M. Elizabeth Sobhia**

Department of Pharmacoinformatics

National Institute of Pharmaceutical Education and Research (NIPER), Mohali
