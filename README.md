# Rags_To_Searches
We implement our own IR system from scratch that supports our custom implemented RAG pipeline to a local LLAMA model.


# Link to book format corpus is derived from.
https://www.ncbi.nlm.nih.gov/books/NBK430685/

# Link to dataset
https://www.kaggle.com/datasets/thedevastator/medmcqa-medical-mcq-dataset

# Local LLM Evaluation (`eval_local_llm.py`)

Evaluates local HuggingFace models on the MedMCQ dataset (`data/50k.csv`).
Supports optional semantic RAG retrieval from the StatPearls corpus via sentence-transformers + FAISS.

```
python eval_local_llm.py --model <hf-model-id> [--rag] [--limit N]
```

## Benchmark Results — 1000 samples, MedMCQ

| Model                              | RAG | Accuracy | Correct / 1000 | Speed    |
|------------------------------------|-----|----------|----------------|----------|
| SmolLM2-135M-Instruct              | No  | 28.10%   | 281            | 44.7 q/s |
| SmolLM2-135M-Instruct              | Yes | 25.00%   | 250            | 7.79 q/s |
| SmolLM2-1.7B-Instruct              | No  | 38.20%   | 382            | 7.09 q/s |
| SmolLM2-1.7B-Instruct              | Yes | 41.70%   | 417            | 1.83 q/s |
| Llama-3.2-1B                       | No  | 29.20%   | 292            | 13.72 q/s |
| Llama-3.2-1B                       | Yes | 14.90%   | 149            | 2.91 q/s |

**Key observations:**
- RAG helps SmolLM2-1.7B (+3.5%) but hurts the smaller 135M and Llama-1B models — larger models better utilise retrieved context
- RAG significantly reduces throughput (3–5× slower) due to encoding overhead
- Random chance baseline is 25% (4-choice MCQ)
