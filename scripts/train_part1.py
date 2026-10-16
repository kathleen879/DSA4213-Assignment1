import time
import torch
import torch.nn as nn
from datasets import load_dataset

from src.dataset import get_dataloaders
from src.utils import set_seed, evaluate_model
from src.models.ngram import NGramLM
from src.models.neural_lms import RNNLM, LSTMLM, TransformerLM

def count_parameters(model):
    """Count trainable parameters for neural models."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_throughput(model, test_loader, device, num_batches=10):
    """Measures inference throughput for neural models in tokens/sec."""
    model.eval()
    model.to(device)
    total_tokens = 0
    
    # Warmup
    with torch.no_grad():
        for i, (x, _) in enumerate(test_loader):
            if i >= 2: break
            _ = model(x.to(device))
            
    start_time = time.time()
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if i >= num_batches: break
            x = x.to(device)
            _ = model(x)
            total_tokens += (y != 0).sum().item()
            
    elapsed = time.time() - start_time
    return total_tokens / elapsed if elapsed > 0 else 0

def train_neural_model(model, train_loader, val_loader, epochs=3, lr=0.001, device="cpu"):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    history = {"train_loss": [], "val_loss": [], "val_ppl": []}
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_non_pad_tokens = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            outputs = model(x)
            loss = criterion(outputs.view(-1, outputs.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            
            non_pad_tokens = (y != 0).sum().item()
            total_loss += loss.item() * non_pad_tokens
            total_non_pad_tokens += non_pad_tokens

        train_loss = total_loss / total_non_pad_tokens if total_non_pad_tokens > 0 else 0
        val_loss, val_ppl = evaluate_model(model, val_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")

    total_time = time.time() - start_time
    return total_time, history

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Setup Data Loaders & Raw Dataset for N-Gram
    train_loader, val_loader, test_loader, vocab = get_dataloaders(seq_len=64, batch_size=32)
    vocab_size = len(vocab)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

# 1. Setup Raw Dataset & Create a CPU-Friendly Subset
    raw_dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    
    # Filter out empty lines first
    train_clean = [t for t in raw_dataset["train"]["text"] if t.strip()]
    val_clean = [t for t in raw_dataset["validation"]["text"] if t.strip()]
    test_clean = [t for t in raw_dataset["test"]["text"] if t.strip()]

    # Select manageable CPU subset sizes
    raw_train_texts = train_clean[:1500]
    raw_val_texts = val_clean[:300]
    raw_test_texts = test_clean[:300]

    # 2. Setup Data Loaders with CPU-optimized settings
    train_loader, val_loader, test_loader, vocab = get_dataloaders(
        seq_len=32,      # Shorter context window speeds up matrix multiplications on CPU
        batch_size=64    # Larger batch size improves parallel processing on CPU cores
    )
    vocab_size = len(vocab)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    results = {}

    # -------------------------------------------------------------
    # 2. Train & Evaluate N-Gram Model
    # -------------------------------------------------------------
    print("\n--- Training 3-Gram LM ---")
    ngram = NGramLM(vocab, n=3, alpha=1.0)
    
    start_time = time.time()
    ngram.fit(raw_train_texts)
    ngram_train_time = time.time() - start_time

    # Evaluate N-Gram on Test set
    ngram_test_loss, ngram_test_ppl = ngram.evaluate_perplexity(raw_test_texts)
    ngram_param_count = len(ngram.counts) if hasattr(ngram, "counts") else 0

    results["3-Gram"] = {
        "params": ngram_param_count,
        "train_time": ngram_train_time,
        "test_loss": ngram_test_loss,
        "test_ppl": ngram_test_ppl,
    }

    # -------------------------------------------------------------
    # 3. Train & Evaluate Neural Models
    # -------------------------------------------------------------
    neural_models = {
        "RNN": RNNLM(vocab_size),
        "LSTM": LSTMLM(vocab_size),
        "Transformer": TransformerLM(vocab_size)
    }

    for name, model in neural_models.items():
        print(f"\n--- Training {name} LM ---")
        params = count_parameters(model)
        train_time, history = train_neural_model(model, train_loader, val_loader, epochs=3, device=device)
        
        test_loss, test_ppl = evaluate_model(model, test_loader, criterion, device)
        
        results[name] = {
            "params": params,
            "train_time": train_time,
            "test_loss": test_loss,
            "test_ppl": test_ppl,
            "history": history
        }
        
    import os
    import pickle

    os.makedirs("checkpoints", exist_ok=True)

    # 1. Save Trigram Model
    with open("checkpoints/trigram_model.pkl", "wb") as f:
        pickle.dump(ngram, f)

    # 2. Save Neural Models
    for name, model in neural_models.items():
        torch.save(model.state_dict(), f"checkpoints/{name.lower()}_model.pt")

    print("\n[SUCCESS] Saved all model checkpoints to 'checkpoints/' folder.")

    # -------------------------------------------------------------
    # 4. Print Complete Comparison Table
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print(f"{'Model':<15} | {'Params / N-Grams':<18} | {'Test Loss':<10} | {'Test PPL':<10} | {'Time (s)':<8}")
    print("="*70)
    for name, res in results.items():
        print(f"{name:<15} | {res['params']:<18} | {res['test_loss']:<10.4f} | {res['test_ppl']:<10.2f} | {res['train_time']:<8.2f}")
    print("="*70)

if __name__ == "__main__":
    main()
    

    
