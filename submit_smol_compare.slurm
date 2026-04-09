#!/bin/bash
#SBATCH --account=clairemcwhite
#SBATCH --job-name=cav_task       # Job name
#SBATCH --output=output_file.txt                # Standard output file (%j = job ID)
#SBATCH --error=new.txt                  # Standard error file (%j = job ID)
#SBATCH --partition=gpu_standard              # Partition name (adjust if necessary)
#SBATCH --ntasks=1                            # Number of tasks
#SBATCH --cpus-per-task=4                     # Number of CPU cores per task (adjust if needed)
#SBATCH --mem=64G                             # Request 64GB of memory
#SBATCH --gpus=1                              # Request 1 GPU
#SBATCH --time=04:00:00                       # 
#SBATCH --mail-type=ALL                       # Mail notifications for all events
#SBATCH --mail-user=shamail@arizona.edu       # Email for notifications
# ---- Environment setup ------------------------------------------------------
conda activate /home/u2/shamail/mihaicrs/llama_eval

export HF_HOME=${SCRATCH:-$HOME}/.hf_cache   # falls back to home if $SCRATCH unset
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p $HF_HOME
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")


echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $(hostname)"
echo "GPUs      : $CUDA_VISIBLE_DEVICES"
echo "Start     : $(date)"
nvidia-smi

# The FAISS index is built once from the corpus and reused by all RAG runs.
FAISS_CACHE="corpus_index.faiss"

# ---- Model 1: SmolLM2-135M-Instruct (baseline + RAG) ------------------------
echo ""
echo "===== Running SmolLM2-135M-Instruct  [no RAG] ====="
python eval_local_llm.py \
    --model "HuggingFaceTB/SmolLM2-135M-Instruct" \
    --data  "data/50k.csv" \
    --output "results_smol135m_1k.csv" \
    --batch-size 32 \
    --limit 1000

echo ""
echo "===== Running SmolLM2-135M-Instruct  [RAG] ====="
python eval_local_llm.py \
    --model "HuggingFaceTB/SmolLM2-135M-Instruct" \
    --data  "data/50k.csv" \
    --output "results_smol135m_1k_rag.csv" \
    --batch-size 32 \
    --limit 1000 \
    --rag --embeddings-cache $FAISS_CACHE

# ---- Model 2: SmolLM2-1.7B-Instruct (baseline + RAG) -----------------------
echo ""
echo "===== Running SmolLM2-1.7B-Instruct  [no RAG] ====="
python eval_local_llm.py \
    --model "HuggingFaceTB/SmolLM2-1.7B-Instruct" \
    --data  "data/50k.csv" \
    --output "results_smol17b_1k.csv" \
    --batch-size 16 \
    --limit 1000

echo ""
echo "===== Running SmolLM2-1.7B-Instruct  [RAG] ====="
python eval_local_llm.py \
    --model "HuggingFaceTB/SmolLM2-1.7B-Instruct" \
    --data  "data/50k.csv" \
    --output "results_smol17b_1k_rag.csv" \
    --batch-size 16 \
    --limit 1000 \
    --rag --embeddings-cache $FAISS_CACHE

# ---- Model 3: Llama-3.2-1B (baseline + RAG) ---------------------------------
echo ""
echo "===== Running Llama-3.2-1B  [no RAG] ====="
python eval_local_llm.py \
    --model "meta-llama/Llama-3.2-1B" \
    --data  "data/50k.csv" \
    --output "results_llama32_1b_1k.csv" \
    --batch-size 16 \
    --limit 1000

echo ""
echo "===== Running Llama-3.2-1B  [RAG] ====="
python eval_local_llm.py \
    --model "meta-llama/Llama-3.2-1B" \
    --data  "data/50k.csv" \
    --output "results_llama32_1b_1k_rag.csv" \
    --batch-size 16 \
    --limit 1000 \
    --rag --embeddings-cache $FAISS_CACHE

# ---- Quick comparison summary -----------------------------------------------
echo ""
echo "===== COMPARISON SUMMARY ====="
python - <<'EOF'
import json, pathlib

for f in [
    "results_smol135m_1k.summary.json",   "results_smol135m_1k_rag.summary.json",
    "results_smol17b_1k.summary.json",    "results_smol17b_1k_rag.summary.json",
    "results_llama32_1b_1k.summary.json", "results_llama32_1b_1k_rag.summary.json",
]:
    p = pathlib.Path(f)
    if p.exists():
        d = json.loads(p.read_text())
        rag_tag = "[RAG]" if d.get("rag") else "[---]"
        print(f"{rag_tag} {d['model']}")
        print(f"  Accuracy : {d['accuracy']:.4f} ({d['accuracy']*100:.2f}%)")
        print(f"  Correct  : {d['correct']} / {d['total']}")
        print(f"  Speed    : {d['questions_per_sec']} q/s")
        print()
EOF

echo "End : $(date)"
