# Rags_To_Searches

Custom IR + RAG pipeline for local LLMs, evaluated on medical MCQs.

- **Corpus**: [StatPearls](https://www.ncbi.nlm.nih.gov/books/NBK430685/)
- **Dataset**: [MedMCQA](https://www.kaggle.com/datasets/thedevastator/medmcqa-medical-mcq-dataset)

## Method

`eval_local_llm.py` runs a local HuggingFace LLM on MedMCQA. With `--rag` it retrieves top-k StatPearls chunks per question via `all-MiniLM-L6-v2` + FAISS and injects them into the prompt.

```bash
python eval_local_llm.py --model <hf-id> [--rag] [--limit N] [--embeddings-cache idx.faiss]
```

## Main Result (1000 questions)

To isolate retrieval vs. fine-tuning, all conditions share the same base architecture: **BioMistral-7B is fine-tuned directly from Mistral-7B-Instruct-v0.1 on PubMed Central**, so the comparison is apples-to-apples.

| Condition | Model | Accuracy |
|---|---|---|
| Base LLM | Mistral-7B-Instruct-v0.1 | 45.9% |
| Base + RAG | Mistral-7B-Instruct-v0.1 | **51.4%** |
| Medical fine-tune | BioMistral-7B | 49.9% |

**Our RAG system beats the medical fine-tune of the same base model** - +5.5pp from retrieval vs +4.0pp from fine-tuning. A small off-the-shelf retrieval system substitutes for (and exceeds) expensive domain pre-training.

The gap is small though - StatPearls is broad but doesn't perfectly match MedMCQA's syllabus (Indian medical exams). A better-aligned corpus would likely widen it. RAG also costs ~20% throughput due to longer prompts.

## Smaller models (1000 questions)

Earlier sweep on small models to establish where RAG starts paying off:

| Model | No RAG | + RAG |
|---|---|---|
| SmolLM2-135M (general, tiny) | 28.1% | 25.0% |
| SmolLM2-1.7B (general, small) | 38.2% | **41.7%** |
| Llama-3.2-1B (base, no instruct) | 29.2% | 14.9% |

Tiny models get *confused* by injected passages rather than helped. RAG only works once the model is (a) large enough to handle long context (~1.7B+) and (b) instruction-tuned to follow "use the references below" directives. Llama-1B's drop from 29% → 15% is a clean example - the base (non-instruct) model treats RAG context as more text to continue rather than as reference material. Random baseline = 25%.

