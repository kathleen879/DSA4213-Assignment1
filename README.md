# DSA4213 Assignment 1: Language Models, Embeddings, and Downstream Representations
## Kathleen Pang, A0282219A

This repository contains the complete implementation and evaluation pipeline for **DSA4213 Assignment 1**, covering $N$-gram and Neural Language Models (RNN, LSTM, Transformer), Word Embedding Ablations (Skip-gram, GloVe), and Downstream Sentiment Analysis on the SST-2 benchmark.

---

## Environment Setup

1. **Clone the repository and navigate to the project root:**
   ```powershell
   git clone https://github.com/kathleen879/DSA4213-Assignment1
   cd Assignment1

2. **Set up virtual environment and install required packages**
    ```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install torch datasets scikit-learn

3. **Train and Evaluate Part 1 Models**
    ```powershell
    python -m scripts.train_part1

4. **Generate Text Continuations**
    ```powershell
    python -m scripts.generate_text

5. **Run Part 2 Script**
    ```powershell
    python -m scripts.run_part2

6. **Run Downstream SST-2 Sentiment Analysis**
    ```powershell
    python -m scripts.run_part3




