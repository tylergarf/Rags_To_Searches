#!/bin/bash
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Save HuggingFace model downloads to the group directory (more disk space)
export HF_HOME=/groups/clairemcwhite/ahmad_workspace/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p $HF_HOME

N=1000
DATA="data/50k.csv"
CACHE="corpus_index.faiss"

# Same base architecture across all conditions for clean comparison:
#   BASE    = mistralai/Mistral-7B-Instruct-v0.1   (general)
#   MEDICAL = BioMistral/BioMistral-7B             (PubMed fine-tune of BASE)
BASE="mistralai/Mistral-7B-Instruct-v0.1"
MED="BioMistral/BioMistral-7B"

# Comment-out any condition you don't want this run.
# Each condition writes:  bench_<prefix>.csv, .summary.json, .detail.json
run() {
    local prefix=$1; shift
    echo ""
    echo "===== $prefix ====="
    python eval_local_llm.py \
        --data  "$DATA" \
        --limit "$N"    \
        --output "bench_${prefix}.csv" \
        "$@"
}

# ---------- Baselines ------------------------------------------------------
# 1) Regular LLM (base, no context)
#run base               --model $BASE --batch-size 1

# 2) Regular LLM + medical RAG (no extra tricks)
#run base_rag           --model $BASE --batch-size 1 --rag --embeddings-cache $CACHE

# 3) Fine-tuned medical LLM (no RAG)
#run medical            --model $MED  --batch-size 1

# Bonus: medical LLM + RAG
# run medical_rag      --model $MED  --batch-size 1 --rag --embeddings-cache $CACHE

# ---------- Summarisation sweep -------------------------------------------
# RAG, but the LLM first summarises retrieved passages (without seeing the
# question) and answers using the summary.
# for MAXTOK in 500 750 1000; do
#     run "base_rag_sum${MAXTOK}" \
#         --model $BASE --batch-size 1 \
#         --rag --embeddings-cache $CACHE \
#         --summarize --summary-max-tokens "$MAXTOK"
# done

# ---------- Chain-of-thought ----------------------------------------------
# CoT lets the model reason step by step before emitting the final letter.
run base_cot           --model $BASE --batch-size 1 --cot
run base_rag_cot       --model $BASE --batch-size 1 --rag --embeddings-cache $CACHE --cot

# ---------- HyDE-style query expansion ------------------------------------
# The LLM first writes a hypothetical answer passage; we retrieve on
# (question + hypothetical) instead of the bare question.
run base_rag_hyde      --model $BASE --batch-size 1 --rag --embeddings-cache $CACHE --query-expand

# ---------- Combos --------------------------------------------------------
# Stacked tricks. Useful to see whether they compose or saturate.
run base_rag_cot_hyde  --model $BASE --batch-size 1 --rag --embeddings-cache $CACHE --cot --query-expand
# run base_rag_sum750_cot --model $BASE --batch-size 1 --rag --embeddings-cache $CACHE --summarize --summary-max-tokens 750 --cot

# ---------- Aggregate all summary.json files into one comparison file -----
echo ""
echo "===== BENCHMARK SUMMARY ($N questions) ====="
python - <<'EOF'
import json, pathlib

# Pretty labels for known conditions; anything not listed falls back to its prefix.
LABELS = {
    "base":               "(1)  Base LLM",
    "base_rag":           "(2)  Base + RAG",
    "medical":            "(3)  Medical fine-tune",
    "medical_rag":        "(4)  Medical FT + RAG",
    "base_rag_sum500":    "(5)  Base+RAG sum 500 tok",
    "base_rag_sum750":    "(6)  Base+RAG sum 750 tok",
    "base_rag_sum1000":   "(7)  Base+RAG sum 1000 tok",
    "base_cot":           "(8)  Base + CoT",
    "base_rag_cot":       "(9)  Base + RAG + CoT",
    "base_rag_hyde":      "(10) Base + RAG + HyDE",
    "base_rag_cot_hyde":  "(11) Base + RAG + CoT + HyDE",
    "base_rag_sum750_cot":"(12) Base + RAG + sum750 + CoT",
}

# Discover every bench_*.summary.json in the cwd (in deterministic order).
summary_files = sorted(pathlib.Path(".").glob("bench_*.summary.json"))

rows = []
for f in summary_files:
    prefix = f.stem.replace("bench_", "").replace(".summary", "")
    try:
        d = json.loads(f.read_text())
    except Exception as e:
        print(f"  [skip] {f}: {e}")
        continue
    rows.append({
        "condition":          prefix,
        "label":              LABELS.get(prefix, prefix),
        "model":              d.get("model", ""),
        "rag":                d.get("rag", False),
        "rag_top_k":          d.get("rag_top_k"),
        "rag_embedder":       d.get("rag_embedder"),
        "summarize":          d.get("summarize", False),
        "summary_max_tokens": d.get("summary_max_tokens"),
        "cot":                d.get("cot", False),
        "cot_max_tokens":     d.get("cot_max_tokens"),
        "query_expand":       d.get("query_expand", False),
        "hyde_max_tokens":    d.get("hyde_max_tokens"),
        "total":              d.get("total"),
        "correct":            d.get("correct"),
        "accuracy":           d.get("accuracy"),
        "elapsed_s":          d.get("elapsed_s"),
        "questions_per_sec":  d.get("questions_per_sec"),
    })

# Sort by accuracy desc for the printed table; keep insertion order in JSON.
rows_sorted = sorted(rows, key=lambda r: (r["accuracy"] or 0), reverse=True)

print(f"{'Condition':<32} {'Model':<32} {'Acc':>7}  {'Correct':>10}  {'q/s':>5}")
print("-" * 95)
for r in rows_sorted:
    model = (r["model"] or "").split("/")[-1][:30]
    acc = f"{(r['accuracy'] or 0)*100:>6.2f}%"
    correct = f"{r['correct']:>4}/{r['total']:<4}" if r["correct"] is not None else ""
    qps = f"{r['questions_per_sec']:>5.1f}" if r["questions_per_sec"] is not None else ""
    print(f"{r['label']:<32} {model:<32} {acc:>7}  {correct:>10}  {qps:>5}")

# Single combined file for downstream comparison / plotting.
out = pathlib.Path("bench_results.json")
out.write_text(json.dumps({
    "n_questions": rows[0]["total"] if rows else None,
    "conditions":  rows,
}, indent=2))
print(f"\nCombined results written to: {out}")
EOF
