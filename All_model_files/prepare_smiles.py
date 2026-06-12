```python
#!/usr/bin/env python3

import argparse
import pickle
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


def main():

    parser = argparse.ArgumentParser(
        description="Prepare SMILES dataset for LSTM training"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV file containing SMILES"
    )

    parser.add_argument(
        "--smiles_column",
        default="smiles",
        help="Column name containing SMILES (default: smiles)"
    )

    parser.add_argument(
        "--mw_cutoff",
        type=float,
        default=500,
        help="Minimum molecular weight cutoff (default: 500)"
    )

    parser.add_argument(
        "--output_csv",
        default="filtered_smiles.csv",
        help="Output filtered SMILES CSV"
    )

    parser.add_argument(
        "--tokenizer",
        default="tokenizer.pkl",
        help="Output tokenizer file"
    )

    args = parser.parse_args()

    print(f"Loading {args.input} ...")

    df = pd.read_csv(args.input)

    if args.smiles_column not in df.columns:
        raise ValueError(
            f"Column '{args.smiles_column}' not found in CSV."
        )

    df[args.smiles_column] = (
        df[args.smiles_column]
        .astype(str)
        .str.strip()
    )

    # =========================
    # Remove duplicates
    # =========================

    total_smiles = len(df)

    num_duplicates = (
        df.duplicated(
            subset=args.smiles_column
        ).sum()
    )

    df_unique = df.drop_duplicates(
        subset=args.smiles_column
    )

    print(f"Total SMILES: {total_smiles}")
    print(f"Removed duplicates: {num_duplicates}")
    print(f"Unique SMILES remaining: {len(df_unique)}")

    # =========================
    # Remove salts
    # =========================

    df_unique = df_unique[
        ~df_unique[
            args.smiles_column
        ].str.contains(r"\.")
    ]

    print(
        f"After removing salts: {len(df_unique)} molecules"
    )

    # =========================
    # MW filtering
    # =========================

    filtered_smiles = []

    for smi in df_unique[args.smiles_column]:

        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            continue

        mw = Descriptors.MolWt(mol)

        if mw > args.mw_cutoff:
            filtered_smiles.append(smi)

    print(
        f"Total molecules with MW > {args.mw_cutoff}: "
        f"{len(filtered_smiles)}"
    )

    # =========================
    # Save filtered SMILES
    # =========================

    pd.DataFrame(
        filtered_smiles,
        columns=["smiles"]
    ).to_csv(
        args.output_csv,
        index=False
    )

    print(
        f"Filtered SMILES saved as "
        f"'{args.output_csv}'"
    )

    # =========================
    # Tokenization
    # =========================

    chars = sorted(
        set(
            "".join(filtered_smiles)
        )
    )

    char2idx = {
        c: i + 1
        for i, c in enumerate(chars)
    }

    char2idx["<PAD>"] = 0

    idx2char = {
        i: c
        for c, i in char2idx.items()
    }

    vocab_size = len(char2idx)

    max_len = max(
        len(s)
        for s in filtered_smiles
    )

    tokenizer = {
        "char2idx": char2idx,
        "idx2char": idx2char,
        "vocab_size": vocab_size,
        "max_len": max_len
    }

    with open(
        args.tokenizer,
        "wb"
    ) as f:

        pickle.dump(
            tokenizer,
            f
        )

    print(
        f"Tokenizer saved as "
        f"'{args.tokenizer}'"
    )

    print("Preprocessing complete")


if __name__ == "__main__":
    main()
```
