import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import Counter

class Vocabulary:
    def __init__(self, pad_token="<pad>", unk_token="<unk>", max_size=15000):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.max_size = max_size
        
        self.w2i = {self.pad_token: 0, self.unk_token: 1}
        self.i2w = {0: self.pad_token, 1: self.unk_token}

    def build_vocab(self, text_list):
        counter = Counter()
        for text in text_list:
            tokens = text.strip().split()
            counter.update(tokens)
            
        # Select top max_size most frequent words
        most_common = counter.most_common(self.max_size - len(self.w2i))
        for word, _ in most_common:
            idx = len(self.w2i)
            self.w2i[word] = idx
            self.i2w[idx] = word

    def encode(self, tokens):
        return [self.w2i.get(token, self.w2i[self.unk_token]) for token in tokens]

    def decode(self, indices):
        return [self.i2w.get(idx, self.unk_token) for idx in indices]

    def __len__(self):
        return len(self.w2i)


class LanguageModelDataset(Dataset):
    def __init__(self, text_list, vocab, seq_len=64):
        self.seq_len = seq_len
        self.vocab = vocab
        
        # Tokenize entire corpus into a single contiguous token sequence
        full_tokens = []
        for text in text_list:
            tokens = text.strip().split()
            if tokens:
                full_tokens.extend(tokens)
                
        # Encode tokens to numerical IDs
        self.encoded = torch.tensor(vocab.encode(full_tokens), dtype=torch.long)

    def __len__(self):
        # Number of fixed sequences available
        return (len(self.encoded) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        
        # Input (x) and Target (y) shifted by 1 token for next-token prediction
        x = self.encoded[start:end]
        y = self.encoded[start + 1:end + 1]
        return x, y


def get_dataloaders(seq_len=64, batch_size=32, vocab_size=15000):
    print("Loading WikiText-2 dataset from Hugging Face...")
    raw_dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    
    train_texts = raw_dataset["train"]["text"]
    val_texts = raw_dataset["validation"]["text"]
    test_texts = raw_dataset["test"]["text"]

    print("Building vocabulary...")
    vocab = Vocabulary(max_size=vocab_size)
    vocab.build_vocab(train_texts)

    print("Creating datasets...")
    train_ds = LanguageModelDataset(train_texts, vocab, seq_len=seq_len)
    val_ds = LanguageModelDataset(val_texts, vocab, seq_len=seq_len)
    test_ds = LanguageModelDataset(test_texts, vocab, seq_len=seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, vocab


if __name__ == "__main__":
    train_loader, val_loader, test_loader, vocab = get_dataloaders()
    print(f"\n--- Dataset Pipeline Status ---")
    print(f"Vocabulary Size: {len(vocab)}")
    print(f"Training Batches: {len(train_loader)}")
    print(f"Validation Batches: {len(val_loader)}")
    print(f"Test Batches: {len(test_loader)}")
    
    x, y = next(iter(train_loader))
    print(f"Sample Batch X shape: {x.shape} (Batch Size x Seq Length)")
    print(f"Sample Batch Y shape: {y.shape}")