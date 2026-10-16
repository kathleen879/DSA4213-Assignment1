import os
import math
import time
import zipfile
import inspect
import urllib.request
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset

from src.models.neural_lms import TransformerLM
from src.utils import set_seed, evaluate_model

# =====================================================================
# 1. Helper to Instantiate TransformerLM with 300d Embeddings
# =====================================================================
def create_transformer_300d(vocab_size):
    """Dynamically detects the dimension parameter name in TransformerLM."""
    sig = inspect.signature(TransformerLM.__init__)
    params = sig.parameters

    kwargs = {}
    if "d_embed" in params:
        kwargs["d_embed"] = 300
    elif "emb_dim" in params:
        kwargs["emb_dim"] = 300
    elif "d_model" in params:
        kwargs["d_model"] = 300
    elif "embed_dim" in params:
        kwargs["embed_dim"] = 300

    try:
        return TransformerLM(vocab_size, **kwargs)
    except TypeError:
        # Fallback if d_model is passed positionally or expects d_model
        return TransformerLM(vocab_size, 300)

# =====================================================================
# 2. Word-Level Vocabulary Builder
# =====================================================================
class WordVocabulary:
    def __init__(self, pad_token="<pad>", unk_token="<unk>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.stoi = {pad_token: 0, unk_token: 1}
        self.itos = {0: pad_token, 1: unk_token}

    def build_vocab(self, text_list, max_size=10000, min_freq=1):
        word_counts = Counter()
        for text in text_list:
            if isinstance(text, str) and text.strip():
                words = text.strip().lower().split()
                word_counts.update(words)

        idx = len(self.stoi)
        for word, count in word_counts.most_common(max_size):
            if count >= min_freq and word not in self.stoi:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def encode(self, text):
        words = text.strip().lower().split() if isinstance(text, str) else text
        return [self.stoi.get(w, self.stoi[self.unk_token]) for w in words]

    def __len__(self):
        return len(self.stoi)

# =====================================================================
# 3. Local Skip-gram Training (Setting B)
# =====================================================================
def train_skipgram_embeddings(texts, vocab, embed_dim=300, window=5, epochs=1, device="cpu"):
    """Trains a simple Skip-gram model on the training text split."""
    print("  -> Training local Skip-gram (Word2Vec) embeddings...")
    vocab_size = len(vocab)
    
    class SkipGramModel(nn.Module):
        def __init__(self, v_size, d):
            super().__init__()
            self.in_embed = nn.Embedding(v_size, d)
            self.out_embed = nn.Linear(d, v_size, bias=False)
        def forward(self, center_words):
            return self.out_embed(self.in_embed(center_words))

    pairs = []
    for text in texts:
        words = text.strip().lower().split()
        ids = [vocab.stoi.get(w, 1) for w in words]
        for i, center in enumerate(ids):
            for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                if i != j:
                    pairs.append((center, ids[j]))
    
    if not pairs:
        return torch.randn(vocab_size, embed_dim)

    # Slice pairs for ultra-fast CPU execution (~30s)
    pairs = pairs[:80000]
    pairs_tensor = torch.tensor(pairs, dtype=torch.long)
    dataset = TensorDataset(pairs_tensor[:, 0], pairs_tensor[:, 1])
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    sg_model = SkipGramModel(vocab_size, embed_dim).to(device)
    optimizer = torch.optim.Adam(sg_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    sg_model.train()
    for _ in range(epochs):
        for center, context in loader:
            center, context = center.to(device), context.to(device)
            optimizer.zero_grad()
            logits = sg_model(center)
            loss = criterion(logits, context)
            loss.backward()
            optimizer.step()

    return sg_model.in_embed.weight.data.cpu()

# =====================================================================
# 4. GloVe Loading & OOV Calculation (Setting C)
# =====================================================================
def load_glove_embeddings(vocab, embed_dim=300):
    """Loads GloVe 6B 300d embeddings and computes OOV coverage."""
    glove_dir = "data/glove"
    glove_path = os.path.join(glove_dir, f"glove.6B.{embed_dim}d.txt")
    os.makedirs(glove_dir, exist_ok=True)

    if not os.path.exists(glove_path):
        zip_path = os.path.join(glove_dir, "glove.6B.zip")
        if not os.path.exists(zip_path):
            print("  -> Downloading GloVe embeddings (glove.6B.zip)...")
            url = "https://nlp.stanford.edu/data/glove.6B.zip"
            urllib.request.urlretrieve(url, zip_path)
        
        print("  -> Extracting GloVe embeddings...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(glove_dir)

    print("  -> Loading GloVe vectors into memory...")
    glove_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = [float(x) for x in parts[1:]]
            if len(vector) == embed_dim:
                glove_dict[word] = vector

    vocab_size = len(vocab)
    weights = torch.zeros(vocab_size, embed_dim)
    found_count = 0

    variance = 0.05
    for word, idx in vocab.stoi.items():
        if word in glove_dict:
            weights[idx] = torch.tensor(glove_dict[word])
            found_count += 1
        else:
            weights[idx] = torch.randn(embed_dim) * math.sqrt(variance)

    oov_coverage = (found_count / vocab_size) * 100
    return weights, oov_coverage

# =====================================================================
# 5. Training Function for Part II
# =====================================================================
def train_setting(model, train_loader, val_loader, test_loader, epochs=2, lr=0.001, device="cpu"):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs.view(-1, outputs.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()

    val_loss, _ = evaluate_model(model, val_loader, criterion, device)
    test_loss, test_ppl = evaluate_model(model, test_loader, criterion, device)
    return val_loss, test_ppl

def set_embedding_weights(model, weights, freeze=True):
    """Utility to set and optionally freeze embedding weights safely."""
    target_layer = None
    if hasattr(model, "embedding"):
        target_layer = model.embedding
    elif hasattr(model, "tok_emb"):
        target_layer = model.tok_emb
    elif hasattr(model, "token_embedding"):
        target_layer = model.token_embedding

    if target_layer is not None:
        if isinstance(target_layer, nn.Embedding):
            target_layer.weight.data.copy_(weights)
            if freeze:
                target_layer.weight.requires_grad = False
        else:
            # If embedding is a sub-module
            target_layer.weight.data.copy_(weights)
            if freeze:
                target_layer.weight.requires_grad = False

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load Raw Text & Build Vocabulary
    raw_dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    train_clean = [t for t in raw_dataset["train"]["text"] if t.strip()]
    val_clean = [t for t in raw_dataset["validation"]["text"] if t.strip()]
    test_clean = [t for t in raw_dataset["test"]["text"] if t.strip()]

    vocab = WordVocabulary()
    vocab.build_vocab(train_clean[:2000], max_size=10000)
    vocab_size = len(vocab)
    print(f"Built Word Vocabulary Size: {vocab_size}")

    # Build Sequence Tensors
    def create_dataset_tensor(texts, seq_len=32):
        all_ids = []
        for t in texts:
            all_ids.extend(vocab.encode(t))
        num_sequences = len(all_ids) // (seq_len + 1)
        if num_sequences == 0:
            return None
        all_ids = all_ids[:num_sequences * (seq_len + 1)]
        data = torch.tensor(all_ids).view(num_sequences, seq_len + 1)
        x, y = data[:, :-1], data[:, 1:]
        return TensorDataset(x, y)

    train_data = create_dataset_tensor(train_clean[:1000], seq_len=32)
    val_data = create_dataset_tensor(val_clean[:300], seq_len=32)
    test_data = create_dataset_tensor(test_clean[:300], seq_len=32)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    results = {}

    # -----------------------------------------------------------------
    # Setting A: Random Joint (Baseline)
    # -----------------------------------------------------------------
    print("\n--- Running Setting A: Random Joint (Trainable) ---")
    model_a = create_transformer_300d(vocab_size)
    val_loss_a, test_ppl_a = train_setting(model_a, train_loader, val_loader, test_loader, epochs=2, device=device)
    results["A: Random Joint"] = {"oov": "100%", "val_loss": val_loss_a, "test_ppl": test_ppl_a}

    # -----------------------------------------------------------------
    # Setting B: Self-Trained Frozen (Skip-gram)
    # -----------------------------------------------------------------
    print("\n--- Running Setting B: Self-Trained Frozen (Skip-gram) ---")
    sg_weights = train_skipgram_embeddings(train_clean[:1000], vocab, embed_dim=300, device=device)
    model_b = create_transformer_300d(vocab_size)
    set_embedding_weights(model_b, sg_weights, freeze=True)

    val_loss_b, test_ppl_b = train_setting(model_b, train_loader, val_loader, test_loader, epochs=2, device=device)
    results["B: Self-Trained (Word2Vec)"] = {"oov": "100%", "val_loss": val_loss_b, "test_ppl": test_ppl_b}

    # -----------------------------------------------------------------
    # Setting C: Public Pretrained Frozen (GloVe)
    # -----------------------------------------------------------------
    print("\n--- Running Setting C: Public Pretrained Frozen (GloVe) ---")
    glove_weights, oov_cov = load_glove_embeddings(vocab, embed_dim=300)
    model_c = create_transformer_300d(vocab_size)
    set_embedding_weights(model_c, glove_weights, freeze=True)

    val_loss_c, test_ppl_c = train_setting(model_c, train_loader, val_loader, test_loader, epochs=2, device=device)
    results["C: Public Pretrained (GloVe)"] = {"oov": f"{oov_cov:.1f}%", "val_loss": val_loss_c, "test_ppl": test_ppl_c}

    # -----------------------------------------------------------------
    # Print Results Table
    # -----------------------------------------------------------------
    print("\n" + "="*70)
    print(f"{'Embedding Setting':<30} | {'OOV Coverage':<12} | {'Val Loss':<10} | {'Test PPL':<10}")
    print("="*70)
    for setting, res in results.items():
        print(f"{setting:<30} | {res['oov']:<12} | {res['val_loss']:<10.4f} | {res['test_ppl']:<10.2f}")
    print("="*70)

if __name__ == "__main__":
    main()