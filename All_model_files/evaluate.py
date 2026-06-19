#!/usr/bin/env python3

import argparse
import pandas as pd

from rdkit import Chem
from rdkit.Chem import (
    Descriptors,
    Lipinski
)

from rdkit.Chem.Scaffolds import (
    MurckoScaffold
)

import sascorer


# =========================================================
# Arguments
# =========================================================

parser = argparse.ArgumentParser(
    description="Evaluate Generated Molecules"
)

parser.add_argument(
    "--input",
    required=True,
    help="CSV containing SMILES"
)

parser.add_argument(
    "--output",
    default="evaluated_molecules.csv",
    help="Output CSV file"
)

args = parser.parse_args()


# =========================================================
# Scaffold
# =========================================================

def get_scaffold(mol):

    try:

        scaffold = (
            MurckoScaffold
            .MurckoScaffoldSmiles(
                mol=mol
            )
        )

        return scaffold

    except:

        return ""



# =========================================================
# Evaluate Molecule
# =========================================================

def evaluate_smiles(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    canonical = Chem.MolToSmiles(
        mol,
        canonical=True
    )

    mw = Descriptors.MolWt(mol)

    logp = Descriptors.MolLogP(mol)

    hba = Lipinski.NumHAcceptors(
        mol
    )

    hbd = Lipinski.NumHDonors(
        mol
    )

    rot_bonds = (
        Lipinski.NumRotatableBonds(
            mol
        )
    )

    chiral_centers = len(
        Chem.FindMolChiralCenters(
            mol,
            includeUnassigned=True
        )
    )

    sa_score = (
        sascorer.calculateScore(
            mol
        )
    )

    scaffold = get_scaffold(
        mol
    )

    return {

        "smiles":
        canonical,

        "scaffold":
        scaffold,

        "sa_score":
        round(sa_score, 3),

        "molecular_weight":
        round(mw, 3),

        "logP":
        round(logp, 3),

        "h_acceptors":
        hba,

        "h_donors":
        hbd,

        "chiral_centers":
        chiral_centers,

        "rotatable_bonds":
        rot_bonds
    }


# =========================================================
# Load Data
# =========================================================

print(
    f"\nLoading molecules:"
)

df = pd.read_csv(
    args.input
)

if "smiles" not in df.columns:

    raise ValueError(
        "Input CSV must contain "
        "'smiles' column"
    )

smiles_list = (
    df["smiles"]
    .astype(str)
    .tolist()
)

print(
    f"Input molecules: "
    f"{len(smiles_list)}"
)


# =========================================================
# Evaluate
# =========================================================

results = []

invalid = 0

for smi in smiles_list:

    res = evaluate_smiles(
        smi
    )

    if res is None:

        invalid += 1

        continue

    results.append(
        res
    )

out_df = pd.DataFrame(
    results
)

# =========================================================
# Save
# =========================================================

out_df.to_csv(
    args.output,
    index=False
)

print(
    f"\nValid molecules : "
    f"{len(out_df)}"
)

print(
    f"Invalid molecules : "
    f"{invalid}"
)

print(
    f"\nSaved results:"
)

print(args.output)