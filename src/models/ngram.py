import torch
import math
from collections import defaultdict, Counter

class NGramLM:
    def __init__(self, vocab, n=3, alpha=1.0):
        """
        n: order of the n-gram (e.g., 3 for Trigram)
        alpha: smoothing parameter for Laplace (Add-alpha) smoothing
        """
        self.n = n
        self.alpha = alpha
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.counts = defaultdict(Counter)
        self.context_counts = Counter()

    def fit(self, text_list):
        """Train the N-gram model by counting sequence frequencies."""
        full_tokens = []
        for text in text_list:
            tokens = text.strip().split() if isinstance(text, str) else text
            if tokens:
                full_tokens.extend(tokens)
                
        # Count n-grams
        for i in range(len(full_tokens) - self.n + 1):
            context = tuple(full_tokens[i : i + self.n - 1])
            target = full_tokens[i + self.n - 1]
            
            self.counts[context][target] += 1
            self.context_counts[context] += 1

    def predict_next_token(self, context_tokens):
        """Predicts the highest-probability next token ID given a context list of token IDs."""
        # Grab the last (n - 1) tokens for context window
        if len(context_tokens) >= self.n - 1:
            context = tuple(context_tokens[-(self.n - 1):])
        else:
            context = tuple(context_tokens)

        # If context exists in counts, pick the most frequent next token
        if context in self.counts and len(self.counts[context]) > 0:
            return self.counts[context].most_common(1)[0][0]
        
        # Fallback if unseen context: pick <unk> token ID (usually 1 or vocab['<unk>'])
        if hasattr(self.vocab, '__getitem__'):
            return self.vocab['<unk>'] if '<unk>' in self.vocab else 1
        return 1

    def evaluate_perplexity(self, text_list):
        """Compute perplexity on evaluation texts with Add-alpha smoothing."""
        full_tokens = []
        for text in text_list:
            tokens = text.strip().split() if isinstance(text, str) else text
            if tokens:
                full_tokens.extend(tokens)

        total_log_prob = 0.0
        total_tokens = 0

        for i in range(len(full_tokens) - self.n + 1):
            context = tuple(full_tokens[i : i + self.n - 1])
            target = full_tokens[i + self.n - 1]

            count = self.counts[context][target]
            context_count = self.context_counts[context]

            # Add-alpha (Laplace) smoothed probability
            prob = (count + self.alpha) / (context_count + self.alpha * self.vocab_size)
            total_log_prob += math.log(prob)
            total_tokens += 1

        if total_tokens == 0:
            return 0.0, float('inf')

        avg_neg_log_likelihood = -total_log_prob / total_tokens
        perplexity = math.exp(avg_neg_log_likelihood)
        return avg_neg_log_likelihood, perplexity