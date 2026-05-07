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
