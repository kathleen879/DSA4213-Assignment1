import os
import pickle
import torch
import torch.nn as nn

from src.dataset import get_dataloaders
from src.utils import set_seed
from src.models.ngram import NGramLM
from src.models.neural_lms import RNNLM, LSTMLM, TransformerLM

def generate_ngram_continuation(ngram_model, prompt_tokens, max_gen_len=30):
    """Generates text continuation using N-Gram model."""
    current_tokens = list(prompt_tokens)
    for _ in range(max_gen_len):
        if hasattr(ngram_model, "predict_next_token"):
            next_token = ngram_model.predict_next_token(current_tokens)
        elif hasattr(ngram_model, "predict"):
            next_token = ngram_model.predict(current_tokens)
        else:
            next_token = 1
        current_tokens.append(next_token)
    return current_tokens

def generate_neural_continuation(model, prompt_tokens, max_gen_len=30, temperature=1.0, device="cpu"):
    """Generates text continuation using Greedy decoding (T=1.0) for Neural Models."""
    model.eval()
    model.to(device)
    
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    generated = list(prompt_tokens)
    
    with torch.no_grad():
        for _ in range(max_gen_len):
            outputs = model(input_ids)
            next_token_logits = outputs[0, -1, :] / temperature
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1).item()
            
            generated.append(next_token)
            
            next_input = torch.tensor([[next_token]], dtype=torch.long, device=device)
            input_ids = torch.cat([input_ids, next_input], dim=1)
            
    return generated

def decode_tokens(token_ids, vocab):
    """Converts token IDs into clean, space-separated word sentences."""
    if hasattr(vocab, "decode"):
        text = vocab.decode(token_ids)
    elif hasattr(vocab, "lookup_tokens"):
        tokens = vocab.lookup_tokens(token_ids)
        text = " ".join(tokens)
    elif isinstance(vocab, dict):
        id_to_token = {v: k for k, v in vocab.items()}
        text = " ".join([id_to_token.get(idx, "<unk>") for idx in token_ids])
    else:
        text = " ".join(str(idx) for idx in token_ids)

    if isinstance(text, list):
        text = " ".join(str(t) for t in text)

    return text.replace(" ,", ",").replace(" .", ".").replace(" <pad>", "").replace("<pad>", "").strip()

def encode_prompt(prompt_text, vocab):
    """Encodes prompt string into word-level token IDs."""
    words = prompt_text.strip().lower().split()
    if hasattr(vocab, "encode"):
        return vocab.encode(words)
    elif hasattr(vocab, "__getitem__"):
        return [vocab[w] if w in vocab else vocab.get("<unk>", 1) for w in words]
    return []

def load_all_models(vocab, checkpoint_dir="checkpoints", device="cpu"):
    """Loads all 4 trained model checkpoints from disk."""
    models = {}
    vocab_size = len(vocab)

    # 1. Load Trigram Model
    trigram_path = os.path.join(checkpoint_dir, "trigram_model.pkl")
    if os.path.exists(trigram_path):
        with open(trigram_path, "rb") as f:
            models["Trigram"] = pickle.load(f)
        print("✓ Loaded Trigram checkpoint.")

    # 2. Load RNN
    rnn = RNNLM(vocab_size)
    rnn_path = os.path.join(checkpoint_dir, "rnn_model.pt")
    if os.path.exists(rnn_path):
        rnn.load_state_dict(torch.load(rnn_path, map_location=device))
        rnn.to(device).eval()
        models["RNN"] = rnn
        print("✓ Loaded RNN checkpoint.")

    # 3. Load LSTM
    lstm = LSTMLM(vocab_size)
    lstm_path = os.path.join(checkpoint_dir, "lstm_model.pt")
    if os.path.exists(lstm_path):
        lstm.load_state_dict(torch.load(lstm_path, map_location=device))
        lstm.to(device).eval()
        models["LSTM"] = lstm
        print("✓ Loaded LSTM checkpoint.")

    # 4. Load Transformer
    transformer = TransformerLM(vocab_size)
    trans_path = os.path.join(checkpoint_dir, "transformer_model.pt")
    if os.path.exists(trans_path):
        transformer.load_state_dict(torch.load(trans_path, map_location=device))
        transformer.to(device).eval()
        models["Transformer"] = transformer
        print("✓ Loaded Transformer checkpoint.")

    return models

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # 1. Load Vocabulary
    _, _, _, vocab = get_dataloaders(seq_len=32, batch_size=32)

    # 2. Load Pre-Trained Model Weights from Checkpoints
    print("--- Loading Pre-Trained Model Checkpoints ---")
    models = load_all_models(vocab, checkpoint_dir="checkpoints", device=device)

    # 3. Define Standard Evaluation Prompts
    prompts = [
        "the discovery of",
        "in recent years",
        "the main purpose of"
    ]

    max_gen_len = 30
    temperature = 1.0

    print("\n" + "="*85)
    print(f"{'QUALITATIVE TEXT GENERATION COMPARISON (30 STEPS, GREEDY T=1.0)':^85}")
    print("="*85)

    for prompt in prompts:
        prompt_ids = encode_prompt(prompt, vocab)
        print(f"\n[PROMPT]: \"{prompt}\"")
        print("-" * 85)

        for name, model in models.items():
            if name == "Trigram":
                gen_ids = generate_ngram_continuation(model, prompt_ids, max_gen_len=max_gen_len)
            else:
                # Corrected signature call (removed extra vocab argument)
                gen_ids = generate_neural_continuation(
                    model, prompt_ids, max_gen_len=max_gen_len, temperature=temperature, device=device
                )
            
            gen_text = decode_tokens(gen_ids, vocab)
            print(f"  {name:<12}: {gen_text}")

    print("\n" + "="*85)

if __name__ == "__main__":
    main()