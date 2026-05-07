#!/bin/bash
# Stage 1 of the validation chain.
#
# Clones princeton-pli/QRHead into a fresh sibling directory, installs it,
# downloads the LME detection input, runs Llama-3.1-8B-Instruct LME detection
# using the upstream code unchanged, and compares the resulting top-16 heads
# against the ground truth published in
#   src/qrretriever/configs/Llama-3.1-8B-Instruct_qr_head_LME.yaml
#
# If this passes, the upstream Princeton implementation reproduces. We then
# proceed to Stage 2: swap the model to Qwen2.5-7B-Instruct (which Princeton
# also supports natively, no code changes) and validate against the Qwen LME
# ranking we already have on qr_detect.
#
# Usage (on a GPU box, from any working directory):
#   bash run_princeton_llama_lme.sh
#
# Override:
#   SKIP_INSTALL=1                  # if Princeton's qrretriever is already installed
#   SKIP_DOWNLOAD=1                 # if the LME data file is already in place
#   PRINCETON_DIR=/path/to/dir      # where to clone Princeton (default: ./princeton_QRHead)
#   QRRETRIEVER_ATTN_IMPLEMENTATION # forwarded; default eager (matches paper)

set -euo pipefail

VALIDATION_DIR="$(cd "$(dirname "$0")" && pwd)"
PRINCETON_DIR="${PRINCETON_DIR:-$(cd "$VALIDATION_DIR/.." && pwd)/princeton_QRHead}"
RESULTS_DIR="$VALIDATION_DIR/results"
mkdir -p "$RESULTS_DIR"

step() {
    echo ""
    echo "============================================"
    echo " $1"
    echo " $(date)"
    echo "============================================"
}

# --- Step 0: pre-flight checks ---
step "0/5: Pre-flight checks"

# 0.a Check HF_TOKEN (Llama-3.1-8B-Instruct is gated on HF)
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set."
    echo "Llama-3.1-8B-Instruct is a gated model on Hugging Face. You need an HF"
    echo "access token with access to meta-llama/Llama-3.1-8B-Instruct."
    echo ""
    echo "To fix:"
    echo "  1. Get a token from https://huggingface.co/settings/tokens"
    echo "  2. Request access at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"
    echo "  3. Run: export HF_TOKEN=hf_..."
    echo "  4. Re-run this script."
    exit 10
fi
echo "HF_TOKEN: set (${#HF_TOKEN} chars)"

# 0.b Check Python + torch + transformers + flash_attn imports
echo ""
echo "Checking Python environment..."
python - <<'PY' || PYEXIT=$?
import sys
errors = []

try:
    import torch
    print(f"  torch        : {torch.__version__} (cuda={torch.cuda.is_available()}, devices={torch.cuda.device_count() if torch.cuda.is_available() else 0})")
    if not torch.cuda.is_available():
        errors.append("torch.cuda.is_available() is False -- this script needs a GPU")
except ImportError as e:
    errors.append(f"torch import failed: {e}")

try:
    import transformers
    print(f"  transformers : {transformers.__version__}")
except ImportError as e:
    errors.append(f"transformers import failed: {e}")

try:
    import flash_attn
    print(f"  flash_attn   : {flash_attn.__version__}")
except ImportError as e:
    errors.append(f"flash_attn import failed: {e}")
    print("  flash_attn   : MISSING")
    print("  Princeton's attn_retriever.py hardcodes attn_implementation='flash_attention_2'.")
    print("  Without flash_attn, model loading will crash. Install with one of:")
    print("    pip install flash-attn --no-build-isolation")
    print("  (compile takes ~15 min; needs CUDA toolkit + nvcc)")
    print("  OR pre-built wheel for your CUDA / torch combo from")
    print("    https://github.com/Dao-AILab/flash-attention/releases")

try:
    import huggingface_hub
    print(f"  huggingface_hub: {huggingface_hub.__version__}")
except ImportError:
    errors.append("huggingface_hub not installed; needed for huggingface-cli download")

if errors:
    print("\nERRORS:")
    for e in errors: print(f"  * {e}")
    sys.exit(1)
PY
if [ "${PYEXIT:-0}" != "0" ]; then
    echo ""
    echo "Pre-flight failed; fix the errors above and re-run."
    exit 11
fi

# 0.c GPU memory: Llama-8B at 115K context needs ~40+ GB VRAM
GPU_MEM_GB=$(python -c "import torch; print(int(torch.cuda.get_device_properties(0).total_memory / 1024**3))" 2>/dev/null || echo 0)
echo "GPU memory   : ${GPU_MEM_GB} GB"
if [ "$GPU_MEM_GB" -gt 0 ] && [ "$GPU_MEM_GB" -lt 40 ]; then
    echo "WARNING: GPU has only ${GPU_MEM_GB} GB. Princeton's unmodified detection"
    echo "  on Llama-8B at LME's ~115K-token contexts likely needs >40 GB."
    echo "  H100 (80GB) or A100 80GB recommended. A100 40GB may OOM."
    echo "  Continuing anyway in 5 seconds; Ctrl-C to abort."
    sleep 5
fi

# --- Step 1: clone Princeton repo (fresh, untouched) ---
step "1/5: Clone princeton-pli/QRHead -> $PRINCETON_DIR"
if [ -d "$PRINCETON_DIR/.git" ]; then
    echo "Princeton clone already exists at $PRINCETON_DIR; pulling latest."
    (cd "$PRINCETON_DIR" && git pull --ff-only)
else
    git clone https://github.com/princeton-pli/QRHead.git "$PRINCETON_DIR"
fi
(cd "$PRINCETON_DIR" && git log -1 --oneline)

# --- Step 2: install Princeton's qrretriever in editable mode ---
if [ "${SKIP_INSTALL:-0}" != "1" ]; then
    step "2/5: pip install -e . (Princeton qrretriever)"
    (cd "$PRINCETON_DIR" && pip install -e . | tail -3)
else
    echo "Skipping install (SKIP_INSTALL=1)."
fi

# --- Step 3: download the LME detection input file ---
LME_DATA="$PRINCETON_DIR/data/longmemeval_data/single-session-user_s.json"
if [ "${SKIP_DOWNLOAD:-0}" != "1" ] && [ ! -f "$LME_DATA" ]; then
    step "3/5: Download LME data"
    (cd "$PRINCETON_DIR" && \
     huggingface-cli download PrincetonPLI/QRHead \
        --repo-type dataset \
        --include "data/longmemeval_data/single-session-user_s.json" \
        --local-dir .)
else
    echo "LME data already present at $LME_DATA (or SKIP_DOWNLOAD=1)."
fi
ls -la "$LME_DATA"

# --- Step 4: run Llama-8B LME detection (upstream, unmodified) ---
LLAMA_OUT="$RESULTS_DIR/princeton_llama_8b_lme.json"
LLAMA_CONFIG="$PRINCETON_DIR/src/qrretriever/configs/Llama-3.1-8B-Instruct_full_head.yaml"

step "4/5: Run Princeton's detect_qrhead_lme.py on Llama-3.1-8B-Instruct"
echo "  Model:   meta-llama/Llama-3.1-8B-Instruct"
echo "  Input:   $LME_DATA"
echo "  Output:  $LLAMA_OUT"
echo "  Config:  $LLAMA_CONFIG"
echo "  Note:    no --truncate_by_space (paper default; Llama 128K window fits LME natively)"
echo ""

(cd "$PRINCETON_DIR" && \
 python exp_scripts/detection/detect_qrhead_lme.py \
    --input_file "$LME_DATA" \
    --output_file "$LLAMA_OUT" \
    --config_or_config_path "$LLAMA_CONFIG")

# --- Step 5: compare top-16 to Princeton's published ground truth ---
GROUND_TRUTH="$PRINCETON_DIR/src/qrretriever/configs/Llama-3.1-8B-Instruct_qr_head_LME.yaml"
step "5/5: Compare top-16 to ground truth"
echo "Ground truth file: $GROUND_TRUTH"
echo "Our run:           $LLAMA_OUT"
echo ""

python "$VALIDATION_DIR/compare_to_ground_truth.py" \
    --our_ranking_json "$LLAMA_OUT" \
    --ground_truth_yaml "$GROUND_TRUTH" \
    --top_k 16

EXIT=$?

echo ""
echo "============================================"
echo " Stage 1 done. Exit code: $EXIT"
if [ $EXIT -eq 0 ]; then
    echo " VALIDATION PASSED. Princeton's Llama-8B LME implementation reproduces."
    echo " Next step: bash run_princeton_qwen_lme.sh"
else
    echo " VALIDATION FAILED. Investigate before proceeding."
fi
echo "============================================"
exit $EXIT
