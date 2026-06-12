# === TOKENIZATION ===
chars = sorted(set("".join(smiles_list)))
char2idx = {c: i + 1 for i, c in enumerate(chars)}  # 0 is reserved for padding
char2idx["<PAD>"] = 0
idx2char = {i: c for c, i in char2idx.items()}

vocab_size = len(char2idx)
max_len = max(len(s) for s in smiles_list)
