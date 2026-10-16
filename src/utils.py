import torch
import random
import numpy as np

def set_seed(seed=42):
    """Ensure exact experiment reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_perplexity(loss):
    """Compute perplexity given average cross-entropy loss."""
    return torch.exp(torch.tensor(loss)).item()

def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluates next-token neural language models on validation/test set.
    Excludes padding tokens via criterion(ignore_index=pad_idx).
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x) # Shape: [Batch, SeqLen, VocabSize]
            
            # Flatten to compute loss over all sequence tokens
            loss = criterion(outputs.view(-1, outputs.size(-1)), y.view(-1))
            
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
            
    avg_loss = total_loss / total_tokens
    ppl = compute_perplexity(avg_loss)
    return avg_loss, ppl