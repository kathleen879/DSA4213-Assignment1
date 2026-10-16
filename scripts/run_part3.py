import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.models.neural_lms import LSTMLM, TransformerLM
from src.utils import set_seed

# =====================================================================
# 1. Simple Word Vocabulary Wrapper
# =====================================================================
class SimpleVocab:
    def __init__(self, texts, max_size=10000):
        from collections import Counter
        counts = Counter()
        for t in texts:
            if isinstance(t, str) and t.strip():
                counts.update(t.strip().lower().split())
        self.stoi = {"<pad>": 0, "<unk>": 1}
        for word, _ in counts.most_common(max_size):
            if word not in self.stoi:
                self.stoi[word] = len(self.stoi)

    def encode(self, text):
        words = text.strip().lower().split() if isinstance(text, str) else text
        return [self.stoi.get(w, 1) for w in words]

    def __len__(self):
        return len(self.stoi)

# =====================================================================
# 2. Representation Extraction Function
# =====================================================================
def extract_representations(model, texts, vocab, seq_len=32, pooling="mean", device="cpu"):
    """Extracts pooled sequence representations from a frozen LM."""
    model.eval()
    model.to(device)
    
    features = []
    with torch.no_grad():
        for text in texts:
            tokens = vocab.encode(text)[:seq_len]
            if not tokens:
                tokens = [1]
            
            # Pad sequence
            pad_len = seq_len - len(tokens)
            input_ids = tokens + [0] * pad_len
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            
            # Forward pass through model embedding / hidden layers
            if hasattr(model, "embedding"):
                embeds = model.embedding(input_tensor)
            elif hasattr(model, "tok_emb"):
                embeds = model.tok_emb(input_tensor)
            else:
                embeds = model(input_tensor)

            # Mask out padding tokens
            mask = (input_tensor != 0).unsqueeze(-1).float()  # [1, seq_len, 1]
            valid_len = mask.sum(dim=1).clamp(min=1)

            if pooling == "mean":
                # Average non-padding token vectors
                pooled = (embeds * mask).sum(dim=1) / valid_len
            elif pooling == "last":
                # Get vector of last non-padding token
                last_idx = int(valid_len.item()) - 1
                pooled = embeds[:, last_idx, :]
            else:
                pooled = embeds.mean(dim=1)

            features.append(pooled.squeeze(0).cpu().numpy())

    return features

# =====================================================================
# 3. Main Execution for Part III
# =====================================================================
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # 1. Load SST-2 Dataset
    print("Loading SST-2 Dataset from Hugging Face...")
    sst2 = load_dataset("stanfordnlp/sst2")
    train_texts = sst2["train"]["sentence"][:2000]  # Fast CPU subset
    train_labels = sst2["train"]["label"][:2000]

    val_texts = sst2["validation"]["sentence"][:500]
    val_labels = sst2["validation"]["label"][:500]

    # 2. Build Vocabulary
    vocab = SimpleVocab(train_texts, max_size=10000)
    vocab_size = len(vocab)
    print(f"Vocabulary Size: {vocab_size}")

    # 3. Instantiate Models
    lstm = LSTMLM(vocab_size).to(device)
    transformer = TransformerLM(vocab_size).to(device)

    experiments = [
        ("LSTM LM", lstm, "last"),
        ("LSTM LM", lstm, "mean"),
        ("Transformer LM", transformer, "mean"),
    ]

    results = []

    print("\n--- Extracting Representations & Training Logistic Regression Classifier ---")
    for name, model, pooling in experiments:
        print(f"Running {name} ({pooling.capitalize()} Pooling)...")
        
        # Extract features
        X_train = extract_representations(model, train_texts, vocab, seq_len=32, pooling=pooling, device=device)
        X_val = extract_representations(model, val_texts, vocab, seq_len=32, pooling=pooling, device=device)

        # Train simple Logistic Regression classifier
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, train_labels)

        # Evaluate
        preds = clf.predict(X_val)
        acc = accuracy_score(val_labels, preds) * 100
        f1 = f1_score(val_labels, preds, average="macro")

        results.append({
            "model": name,
            "pooling": "Last Hidden State" if pooling == "last" else "Mean Pooling",
            "acc": acc,
            "f1": f1
        })

    # 4. Print Summary Results Table
    print("\n" + "="*65)
    print(f"{'Source Model':<18} | {'Pooling Strategy':<20} | {'Accuracy (%)':<12} | {'Macro-F1':<8}")
    print("="*85)
    for res in results:
        print(f"{res['model']:<18} | {res['pooling']:<20} | {res['acc']:<12.2f} | {res['f1']:<8.4f}")
    print("="*65)

if __name__ == "__main__":
    main()