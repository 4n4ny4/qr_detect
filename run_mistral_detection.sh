#!/bin/bash
# End-to-end Mistral-7B-Instruct-v0.3 QRHead detection pipeline for
# LongMemEval `single-session-user_s` and BEIR `nq_train`.
#
# Designed to be run from the repo root on a GPU box with a working
# Python env (torch + transformers + huggingface_hub + pyyaml installed).
#
# Steps:
#   1. (optional) pip install -e .
#   2. Verify Mistral-specific imports.
#   3. Download LME and NQ data from PrincetonPLI/QRHead HF dataset.
#   4. Run probe to pick `--truncate_by_space N` for LME.
#   5. Smoke-test Mistral on 2 LME examples / 3 paragraphs.
#   6. Run Mistral detection on LME (using probe-chosen N) and NQ (N=400).
#   7. Write a README to `results/` recording the run config.
#
# LME uses --evidence_preserving_truncation (commit 8d9686e). Gold rounds
# (gt_docs / is_supporting) are kept full; only distractor rounds are
# truncated to N words. NQ uses uniform --truncate_by_space 400 (matches
# Princeton paper / collaborator's Qwen + OLMo runs).
#
# Override behaviour with env vars:
#   SKIP_INSTALL=1                       # skip `pip install -e .`
#   SKIP_DOWNLOAD=1                      # don't re-download data
#   SKIP_SMOKE=1                         # skip the 2-example smoke test
#   FORCE_LME_N=<int>                    # bypass probe; use this N for distractor truncation
#   GOLD_TRUNCATE_BY_SPACE=<int>         # cap on gold rounds (default 0 = preserve fully)
#   DISABLE_EVIDENCE_PRESERVING=1        # fall back to uniform truncation (NOT recommended)
#   QRRETRIEVER_ATTN_IMPLEMENTATION      # forwarded to qrretriever (default: sdpa)
#   QRRETRIEVER_MLP_CHUNK_SIZE           # forwarded to qrretriever (default: 2048)
#   QRRETRIEVER_PREFILL_CHUNK_SIZE       # forwarded to qrretriever (default: 1024)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

LME_DATA="data/longmemeval_data/single-session-user_s.json"
NQ_DATA="data/beir_data/nq_train.json"

CONFIG="src/qrretriever/configs/Mistral-7B-Instruct-v0.3_full_head.yaml"
RESULTS_DIR="results"
PROBE_REPORT="$RESULTS_DIR/probe_report.txt"
LME_OUT="$RESULTS_DIR/mistral_lme.json"
NQ_OUT="$RESULTS_DIR/mistral_nq.json"
README="$RESULTS_DIR/README.md"

mkdir -p "$RESULTS_DIR"

step() {
    echo ""
    echo "============================================"
    echo " $1"
    echo " $(date)"
    echo "============================================"
}

# --- Step 1: install (optional) ---
if [ "${SKIP_INSTALL:-0}" != "1" ]; then
    step "1/6: pip install -e ."
    pip install -e . | tail -5
else
    echo "Skipping install (SKIP_INSTALL=1)."
fi

# --- Step 2: verify imports ---
step "2/6: Verify Mistral imports"
python - <<'PY'
import sys, transformers
print(f"transformers version: {transformers.__version__}")
try:
    from transformers.models.mistral.modeling_mistral import (
        ALL_ATTENTION_FUNCTIONS,
        MistralForCausalLM,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )
    print("Imported ALL_ATTENTION_FUNCTIONS, eager_attention_forward from modeling_mistral.")
except ImportError as e:
    print(f"modeling_mistral does not re-export AttentionInterface symbols: {e}")
    print("Will use fallback imports inside custom_modeling_mistral.py.")
from qrretriever.custom_modeling_mistral import MistralForCausalLM as QRMistral
print(f"qrretriever.custom_modeling_mistral.MistralForCausalLM: {QRMistral}")
print("Mistral imports OK.")
PY

# --- Step 3: download data ---
if [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
    step "3/6: Download LME + NQ data from PrincetonPLI/QRHead"
    if [ ! -f "$LME_DATA" ] || [ ! -f "$NQ_DATA" ]; then
        huggingface-cli download PrincetonPLI/QRHead \
            --repo-type dataset \
            --include "data/longmemeval_data/single-session-user_s.json" \
                      "data/beir_data/nq_train.json" \
            --local-dir .
    else
        echo "Both data files already present; skipping download."
    fi
else
    echo "Skipping download (SKIP_DOWNLOAD=1)."
fi

# --- Step 4: probe + decide N ---
if [ -n "${FORCE_LME_N:-}" ]; then
    LME_N="$FORCE_LME_N"
    DECISION="FORCED_N=$LME_N (FORCE_LME_N override; probe skipped)"
    echo "Skipping probe; using FORCE_LME_N=$LME_N."
else
    step "4/6: Probe LME context lengths against Mistral"
    python exp_scripts/detection/probe_lme_lengths_mistral.py \
        --input_file "$LME_DATA" \
        --output_report "$PROBE_REPORT"

    # Extract decision line from report.
    DECISION_LINE="$(grep '^DECISION:' "$PROBE_REPORT" | head -1)"
    echo "Probe says: $DECISION_LINE"
    if [[ "$DECISION_LINE" == *"SKIP_MISTRAL_LME"* ]]; then
        LME_N=""
        DECISION="SKIP_MISTRAL_LME"
    else
        LME_N="$(echo "$DECISION_LINE" | sed -n 's/.*chosen_N=\([0-9]*\).*/\1/p')"
        if [ -z "$LME_N" ]; then
            echo "ERROR: could not parse chosen_N from probe report; aborting."
            exit 3
        fi
        DECISION="probe-chosen N=$LME_N"
    fi
fi

# --- Step 5: smoke test ---
if [ "${SKIP_SMOKE:-0}" != "1" ]; then
    step "5/6: Smoke test on 2 LME examples"
    python exp_scripts/detection/smoke_test_mistral.py --input_file "$LME_DATA"
else
    echo "Skipping smoke test (SKIP_SMOKE=1)."
fi

# --- Step 6: detection runs ---
step "6/6: Mistral detection runs"

GOLD_TRUNCATE="${GOLD_TRUNCATE_BY_SPACE:-0}"

if [ -n "$LME_N" ]; then
    if [ "${DISABLE_EVIDENCE_PRESERVING:-0}" = "1" ]; then
        echo ">>> LME with UNIFORM --truncate_by_space $LME_N (evidence preservation disabled)"
        python exp_scripts/detection/detect_qrhead_lme.py \
            --input_file "$LME_DATA" \
            --output_file "$LME_OUT" \
            --truncate_by_space "$LME_N" \
            --config_or_config_path "$CONFIG"
        LME_TRUNC_MODE="uniform N=$LME_N"
    else
        echo ">>> LME with EVIDENCE-PRESERVING truncation: distractor N=$LME_N, gold cap=$GOLD_TRUNCATE (0=preserve full)"
        python exp_scripts/detection/detect_qrhead_lme.py \
            --input_file "$LME_DATA" \
            --output_file "$LME_OUT" \
            --truncate_by_space "$LME_N" \
            --evidence_preserving_truncation \
            --gold_truncate_by_space "$GOLD_TRUNCATE" \
            --config_or_config_path "$CONFIG"
        LME_TRUNC_MODE="evidence-preserving (distractor N=$LME_N, gold cap=$GOLD_TRUNCATE)"
    fi
else
    echo ">>> Skipping Mistral-LME per probe decision."
    LME_TRUNC_MODE="N/A (skipped)"
fi

echo ""
echo ">>> NQ with --truncate_by_space 400 (matches Princeton paper default)"
python exp_scripts/detection/detect_qrhead_beir.py \
    --input_file "$NQ_DATA" \
    --output_file "$NQ_OUT" \
    --truncate_by_space 400 \
    --config_or_config_path "$CONFIG"

# --- Write results README ---
TRANSFORMERS_VERSION="$(python -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo unknown)"
TORCH_VERSION="$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo unknown)"
ATTN_IMPL="${QRRETRIEVER_ATTN_IMPLEMENTATION:-sdpa (default)}"
MLP_CHUNK="${QRRETRIEVER_MLP_CHUNK_SIZE:-2048 (default)}"
LME_DATA_SHA="$(shasum -a 256 "$LME_DATA" 2>/dev/null | awk '{print $1}' || echo unknown)"
NQ_DATA_SHA="$(shasum -a 256 "$NQ_DATA" 2>/dev/null | awk '{print $1}' || echo unknown)"

cat > "$README" <<EOF
# Mistral-7B-Instruct-v0.3 QRHead Detection Results

Run date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

## Decision

$DECISION

## Artifacts

- \`probe_report.txt\` - probe output: token-length table, gold-round
  word-count distribution, and the explicit decision rule's chosen N.
$( [ -n "$LME_N" ] && echo "- \`mistral_lme.json\` - QRScore ranking on LongMemEval \`single-session-user_s.json\` with \`--truncate_by_space $LME_N\`." )
- \`mistral_nq.json\` - QRScore ranking on BEIR \`nq_train.json\` with \`--truncate_by_space 400\`.

## Run config

| | |
|---|---|
| Model | mistralai/Mistral-7B-Instruct-v0.3 |
| transformers | $TRANSFORMERS_VERSION |
| torch | $TORCH_VERSION |
| QRRETRIEVER_ATTN_IMPLEMENTATION | $ATTN_IMPL |
| QRRETRIEVER_MLP_CHUNK_SIZE | $MLP_CHUNK |
| LME data | \`$LME_DATA\` (sha256 \`$LME_DATA_SHA\`) |
| NQ data | \`$NQ_DATA\` (sha256 \`$NQ_DATA_SHA\`) |
| LME truncation mode | $LME_TRUNC_MODE |
| NQ truncate_by_space | 400 (uniform; matches paper / collaborator) |
| NQ shuffle seed | 42 (set in \`detect_qrhead_beir.py\`) |
| LME instances run | $( [ -n "$LME_N" ] && echo 70 || echo 0 ) |
| NQ instances run | 128 |

## Reproducing

\`\`\`bash
# from repo root, with a working Python env
bash run_mistral_detection.sh
\`\`\`

Override the probe by setting \`FORCE_LME_N=<int>\`. Skip steps with
\`SKIP_INSTALL=1\`, \`SKIP_DOWNLOAD=1\`, \`SKIP_SMOKE=1\`.

## Cross-model coordination

Qwen and OLMo runs are produced separately by a collaborator. For
strict LME cross-model comparability, share \`probe_report.txt\` with
them and ask whether they will use the same \`--truncate_by_space\` and
\`--evidence_preserving_truncation\` settings. The OLMo run already used
\`--truncate_by_space 5 --evidence_preserving_truncation\`; Mistral can
use a much larger N because of its 32K window. NQ runs are comparable
across models because all use \`--truncate_by_space 400\` and
\`detect_qrhead_beir.py\`'s \`random.seed(42)\` shuffle is deterministic.
EOF

echo ""
echo "============================================"
echo " Pipeline complete."
echo " Results in: $RESULTS_DIR/"
echo " Decision: $DECISION"
echo "============================================"
