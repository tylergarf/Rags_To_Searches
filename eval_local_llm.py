"""
eval_local_llm.py
-----------------
Evaluate a local LLM (via HuggingFace Transformers) on the MedMCQ 50k.csv
dataset.  Optionally augments each question with retrieved context from the
StatPearls corpus using semantic (dense) retrieval (RAG).

Usage examples
--------------
# Baseline – no RAG
python eval_local_llm.py --model HuggingFaceTB/SmolLM2-1.7B-Instruct --limit 500

# With RAG – retrieves top-3 corpus chunks per question
python eval_local_llm.py --model HuggingFaceTB/SmolLM2-1.7B-Instruct --rag --limit 500

# RAG with custom settings
python eval_local_llm.py --rag --top-k 5 --corpus-dir corpus/statpearls/chunk

# Pre-build and cache the corpus embeddings (reused on every subsequent run)
python eval_local_llm.py --rag --embeddings-cache corpus_index.faiss

# 4-bit quantisation for large models
python eval_local_llm.py --model mistralai/Mistral-7B-Instruct-v0.3 --load-in-4bit --rag

# Resume a previously interrupted run
python eval_local_llm.py --rag --output results_rag.csv --resume
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "microsoft/Phi-3.5-mini-instruct"
DEFAULT_DATA = "data/50k.csv"
DEFAULT_OUTPUT = "results_local_llm.csv"
DEFAULT_BATCH = 16
DEFAULT_CORPUS_DIR = "corpus/statpearls/chunk"
DEFAULT_EMBEDDER = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 3
MAX_CTX_CHARS = 1200   # max characters per retrieved chunk shown in the prompt


# ---------------------------------------------------------------------------
# Semantic Retriever (FAISS + sentence-transformers)
# ---------------------------------------------------------------------------
class SemanticRetriever:
    """
    Loads every JSONL chunk from corpus_dir, encodes them with a
    sentence-transformer, stores them in a FAISS flat index, and exposes a
    retrieve(query) method that returns the top-k most relevant passages.
    """

    def __init__(
        self,
        corpus_dir: str,
        embedder_name: str = DEFAULT_EMBEDDER,
        top_k: int = DEFAULT_TOP_K,
        cache_path: str | None = None,
    ):
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "RAG requires extra packages:\n"
                "  pip install sentence-transformers faiss-cpu"
            )

        self.top_k = top_k
        self._faiss = faiss
        self.embedder = SentenceTransformer(embedder_name)

        # ---- Load corpus chunks ---------------------------------------------
        chunks = self._load_corpus(corpus_dir)
        self.texts = [c["content"] for c in chunks]
        self.titles = [c.get("title", "") for c in chunks]
        print(f"[RAG] Corpus: {len(self.texts):,} chunks from {corpus_dir}")

        # ---- Build / load FAISS index ----------------------------------------
        cache = Path(cache_path) if cache_path else None
        emb_cache = cache.with_suffix(".npy") if cache else None

        if emb_cache and emb_cache.exists() and cache and cache.exists():
            print(f"[RAG] Loading cached embeddings from {emb_cache}")
            embeddings = np.load(str(emb_cache)).astype("float32")
            self.index = faiss.read_index(str(cache))
        else:
            print(f"[RAG] Encoding {len(self.texts):,} chunks with '{embedder_name}' …")
            embeddings = self.embedder.encode(
                self.texts,
                batch_size=256,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")

            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)   # inner product = cosine (normalised)
            self.index.add(embeddings)

            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                faiss.write_index(self.index, str(cache))
                np.save(str(emb_cache), embeddings)
                print(f"[RAG] Index saved to {cache}")

        print(f"[RAG] Index ready ({self.index.ntotal:,} vectors, dim={embeddings.shape[1]})")

    @staticmethod
    def _load_corpus(corpus_dir: str) -> list[dict]:
        chunks = []
        for path in sorted(Path(corpus_dir).rglob("*.jsonl")):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        chunks.append(json.loads(line))
        return chunks

    def retrieve(self, query: str) -> list[str]:
        """Return top-k passage strings for a query."""
        q_emb = self.embedder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        _, indices = self.index.search(q_emb, self.top_k)
        return [self.texts[i][:MAX_CTX_CHARS] for i in indices[0] if i >= 0]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_BASE = (
    "You are a medical expert answering multiple-choice questions. "
    "Respond with ONLY the letter of the correct answer (A, B, C, or D). "
    "Do not add any explanation."
)

SYSTEM_PROMPT_RAG = (
    "You are a medical expert answering multiple-choice questions. "
    "Use the reference passages below to help choose the correct answer. "
    "Respond with ONLY the letter of the correct answer (A, B, C, or D). "
    "Do not add any explanation."
)


def build_user_message(row: pd.Series, context_passages: list[str] | None = None) -> str:
    parts = []
    if context_passages:
        ctx = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(context_passages))
        parts.append(f"Reference passages:\n{ctx}\n")
    parts.append(
        f"Question: {row['question']}\n\n"
        f"A. {row['opa']}\n"
        f"B. {row['opb']}\n"
        f"C. {row['opc']}\n"
        f"D. {row['opd']}\n\n"
        "Answer:"
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
COP_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


def ground_truth_letter(cop) -> str:
    return COP_MAP.get(int(cop), "")


def extract_letter(text: str) -> str:
    for ch in text.upper():
        if ch in "ABCD":
            return ch
    return ""


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_name: str, load_in_4bit: bool, load_in_8bit: bool):
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if load_in_4bit or load_in_8bit:
        bnb_ok = False
        try:
            import bitsandbytes  # noqa: F401
            # transformers also does its own check – probe it now
            from transformers.quantizers.quantizers_utils import is_bitsandbytes_available
            bnb_ok = is_bitsandbytes_available()
        except Exception:
            try:
                from transformers.utils import is_bitsandbytes_available
                bnb_ok = is_bitsandbytes_available()
            except Exception:
                bnb_ok = False

        if bnb_ok:
            if load_in_4bit:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            print("WARNING: bitsandbytes not functional – falling back to full bf16 (use smaller --batch-size)")
            load_in_4bit = load_in_8bit = False

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Loading model: {model_name}  (dtype={dtype}, 4bit={load_in_4bit}, 8bit={load_in_8bit})")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        dtype=None if quant_config else dtype,
        device_map="auto",
        trust_remote_code=False,
    )
    model.eval()

    device = next(model.parameters()).device
    print(f"Model loaded on: {device}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  ({mem:.1f} GB)")

    return tokenizer, model


# ---------------------------------------------------------------------------
# Batched inference
# ---------------------------------------------------------------------------
def build_chat_prompt(
    tokenizer,
    row: pd.Series,
    context_passages: list[str] | None = None,
) -> str:
    system = SYSTEM_PROMPT_RAG if context_passages else SYSTEM_PROMPT_BASE
    user_msg = build_user_message(row, context_passages)

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        # Try with a system role first; fall back to merging into user msg
        # for models (e.g. Mistral) whose template doesn't support system.
        try:
            return tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": f"{system}\n\n{user_msg}"}],
                tokenize=False,
                add_generation_prompt=True,
            )
    return f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user_msg} [/INST]"


def run_batch(tokenizer, model, prompts: list[str], max_new_tokens: int = 8) -> list[str]:
    # RAG prompts are longer – allow up to 2048 input tokens
    max_input = 2048
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    return tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Benchmark report
# ---------------------------------------------------------------------------
def print_report(df_results: pd.DataFrame, rag: bool) -> None:
    total = len(df_results)
    correct = (df_results["predicted"] == df_results["ground_truth"]).sum()
    accuracy = correct / total if total else 0.0

    print("\n" + "=" * 60)
    print(f"BENCHMARK RESULTS  (RAG={'ON' if rag else 'OFF'})")
    print("=" * 60)
    print(f"Total questions  : {total}")
    print(f"Correct          : {correct}")
    print(f"Accuracy         : {accuracy:.4f}  ({accuracy*100:.2f}%)")

    if "subject_name" in df_results.columns:
        print("\nPer-subject accuracy (top 15 by question count):")
        sub = (
            df_results.groupby("subject_name")
            .apply(lambda g: pd.Series({
                "n": len(g),
                "acc": (g["predicted"] == g["ground_truth"]).mean(),
            }))
            .sort_values("n", ascending=False)
            .head(15)
        )
        print(sub.to_string())

    abstain = (df_results["predicted"] == "").sum()
    print(f"\nAbstention rate  : {abstain/total:.4f}  ({abstain} rows)")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a local LLM on MedMCQ – with optional RAG"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only first N rows (for quick tests)")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Skip rows already present in --output")
    parser.add_argument("--max-new-tokens", type=int, default=8)

    # RAG arguments
    rag = parser.add_argument_group("RAG (semantic retrieval)")
    rag.add_argument("--rag", action="store_true",
                     help="Enable RAG: retrieve corpus context for each question")
    rag.add_argument("--corpus-dir", default=DEFAULT_CORPUS_DIR,
                     help=f"Directory of JSONL corpus chunks (default: {DEFAULT_CORPUS_DIR})")
    rag.add_argument("--embedder", default=DEFAULT_EMBEDDER,
                     help=f"Sentence-transformer model for retrieval (default: {DEFAULT_EMBEDDER})")
    rag.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                     help=f"Number of passages to retrieve per question (default: {DEFAULT_TOP_K})")
    rag.add_argument("--embeddings-cache", default=None,
                     help="Path to save/load the FAISS index (e.g. corpus_index.faiss). "
                          "Saves significant time on repeated runs.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # ---- Load dataset -------------------------------------------------------
    print(f"Loading dataset: {args.data}")
    df = pd.read_csv(args.data)
    df = df[df["choice_type"] == "single"].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    print(f"Rows to evaluate: {len(df)}")

    # ---- Resume -------------------------------------------------------------
    done_ids: set = set()
    if args.resume and Path(args.output).exists():
        done_df = pd.read_csv(args.output)
        done_ids = set(done_df["id"].astype(str))
        print(f"Resuming – skipping {len(done_ids)} already-evaluated rows")
        df = df[~df["id"].astype(str).isin(done_ids)].reset_index(drop=True)

    if len(df) == 0:
        print("Nothing left to evaluate.")
        return

    # ---- Build retriever (if RAG enabled) -----------------------------------
    retriever: SemanticRetriever | None = None
    if args.rag:
        retriever = SemanticRetriever(
            corpus_dir=args.corpus_dir,
            embedder_name=args.embedder,
            top_k=args.top_k,
            cache_path=args.embeddings_cache,
        )

    # ---- Load LLM -----------------------------------------------------------
    tokenizer, model = load_model(args.model, args.load_in_4bit, args.load_in_8bit)

    # ---- Inference loop -----------------------------------------------------
    output_path = Path(args.output)
    write_header = not (args.resume and output_path.exists())

    fieldnames = ["id", "question", "ground_truth", "predicted", "raw_output",
                  "correct", "rag_context"]
    results = []
    start_time = time.time()

    with open(output_path, "a", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for batch_start in tqdm(
            range(0, len(df), args.batch_size),
            desc="Evaluating",
            unit="batch",
        ):
            batch = df.iloc[batch_start : batch_start + args.batch_size]

            # Retrieve context for each row (fast – CPU only)
            contexts: list[list[str] | None] = []
            for _, row in batch.iterrows():
                if retriever:
                    contexts.append(retriever.retrieve(row["question"]))
                else:
                    contexts.append(None)

            prompts = [
                build_chat_prompt(tokenizer, row, ctx)
                for (_, row), ctx in zip(batch.iterrows(), contexts)
            ]
            raw_outputs = run_batch(tokenizer, model, prompts, args.max_new_tokens)

            for (_, row), raw, ctx in zip(batch.iterrows(), raw_outputs, contexts):
                predicted = extract_letter(raw)
                gt = ground_truth_letter(row["cop"])
                csv_record = {
                    "id": row["id"],
                    "question": row["question"],
                    "ground_truth": gt,
                    "predicted": predicted,
                    "raw_output": raw.strip(),
                    "correct": predicted == gt,
                    "rag_context": " ||| ".join(ctx) if ctx else "",
                }
                writer.writerow(csv_record)
                # Richer record for the JSON dump
                results.append({
                    "id": row["id"],
                    "question": row["question"],
                    "options": {
                        "A": row["opa"], "B": row["opb"],
                        "C": row["opc"], "D": row["opd"],
                    },
                    "ground_truth": gt,
                    "predicted": predicted,
                    "raw_output": raw.strip(),
                    "correct": predicted == gt,
                    "rag_context": ctx if ctx else [],
                })

        fout.flush()

    elapsed = time.time() - start_time
    qps = len(results) / elapsed if elapsed > 0 else 0
    print(f"\nInference complete: {len(results)} questions in {elapsed:.1f}s  ({qps:.1f} q/s)")

    # ---- Report -------------------------------------------------------------
    results_df = pd.read_csv(output_path)
    id_to_subject = (
        df.set_index("id")["subject_name"].to_dict()
        if "subject_name" in df.columns else {}
    )
    results_df["subject_name"] = results_df["id"].map(id_to_subject)
    print_report(results_df, rag=args.rag)

    summary = {
        "model": args.model,
        "data": args.data,
        "rag": args.rag,
        "rag_embedder": args.embedder if args.rag else None,
        "rag_top_k": args.top_k if args.rag else None,
        "total": len(results_df),
        "correct": int((results_df["predicted"] == results_df["ground_truth"]).sum()),
        "accuracy": float((results_df["predicted"] == results_df["ground_truth"]).mean()),
        "elapsed_s": round(elapsed, 1),
        "questions_per_sec": round(qps, 2),
    }
    summary_path = output_path.with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Per-question detail JSON (question, answer, RAG context)
    detail_path = output_path.with_suffix(".detail.json")
    with open(detail_path, "w") as f:
        json.dump({"meta": summary, "questions": results}, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to : {output_path}")
    print(f"Summary saved to : {summary_path}")
    print(f"Detail saved to  : {detail_path}")


if __name__ == "__main__":
    main()
