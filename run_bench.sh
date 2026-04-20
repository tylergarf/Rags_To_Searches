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

# Same base architecture across all 3 conditions for clean comparison:
#   BASE    = mistralai/Mistral-7B-Instruct-v0.1   (general)
#   MEDICAL = BioMistral/BioMistral-7B             (PubMed fine-tune of BASE)
BASE="mistralai/Mistral-7B-Instruct-v0.1"
MED="BioMistral/BioMistral-7B"

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

# 1) Regular LLM (base, no context)
run base          --model $BASE --batch-size 2

# 2) Regular LLM + medical RAG
run base_rag      --model $BASE --batch-size 1 --rag --embeddings-cache $CACHE

# 3) Fine-tuned medical LLM (no RAG)
run medical       --model $MED  --batch-size 2

# Bonus: medical LLM + RAG (does fine-tuning + RAG stack?)
# run medical_rag   --model $MED  --batch-size 1 --rag --embeddings-cache $CACHE

# ---- Summary ----------------------------------------------------------------
echo ""
echo "===== BENCHMARK SUMMARY ($N questions) ====="
python - <<'EOF'
import json, pathlib

label = {
    "base":        "(1) Base LLM            ",
    "base_rag":    "(2) Base LLM + RAG      ",
    "medical":     "(3) Medical fine-tune   ",
    "medical_rag": "(4) Medical FT + RAG    ",
}
print(f"{'Condition':<28} {'Model':<32} {'Accuracy':>9} {'Correct':>10}")
print("-" * 85)
for key in ["base", "base_rag", "medical", "medical_rag"]:
    f = pathlib.Path(f"bench_{key}.summary.json")
    if not f.exists():
        continue
    d = json.loads(f.read_text())
    model = d["model"].split("/")[-1][:30]
    print(f"{label[key]:<28} {model:<32} {d['accuracy']*100:>8.1f}% {d['correct']:>4}/{d['total']:<4}")
EOF
