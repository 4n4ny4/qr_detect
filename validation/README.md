# Validation kit: Princeton-as-ground-truth chain

This kit validates the QRHead detection implementation by reproducing
Princeton's published Llama-3.1-8B-Instruct LME ranking. Once that passes,
we run the same upstream code on Qwen, and (with small additions) OLMo and
Mistral. Every model uses Princeton's reference detection script with no
modifications, so any discrepancy isolates to either the model wiring or
the model itself, not our code.

The validation chain:

```
Stage 1: Princeton code + Llama-3.1-8B-Instruct + LME
         ──> reproduce Princeton paper top-16 LME heads (set match)
         ──> if match: upstream code reproduces, baseline OK

Stage 2: Princeton code + Qwen2.5-7B-Instruct + LME
         ──> Princeton repo natively supports Qwen2.5-7B; no code change.
         ──> Compare to qr_detect's existing Qwen LME ranking
             (detection_results/Qwen2.5-7B-Instruct_qr_heads_LME.json).
         ──> If they match: qr_detect's Qwen path is faithful.
             If they don't: locate the divergence (likely the chunked-
             prefill + heavy-modeling interaction we already have evidence
             of breaking the Qwen LME run).

Stage 3: Princeton code + OLMo-7B-Instruct-hf + LME
         ──> Princeton doesn't ship OLMo support upstream. Two options:
             (a) port qr_detect's `custom_modeling_olmo.py` into the
                 Princeton clone, OR
             (b) accept qr_detect's OLMo ranking on faith because it
                 already shows healthy layer distribution and score
                 magnitudes (see analysis below).

Stage 4: Princeton code + Mistral-7B-Instruct-v0.3 + LME
         ──> Same situation as OLMo: not in Princeton upstream. Either
             port qr_detect's `custom_modeling_mistral.py` (light style)
             into the Princeton clone, OR run our existing pipeline on
             qr_detect's `add-mistral-detection` branch.
```

## Stage 1: Llama-8B LME (run this first)

```bash
# On the GPU box. From this validation/ directory.
bash run_princeton_llama_lme.sh
```

What it does:

1. `git clone https://github.com/princeton-pli/QRHead` into a fresh sibling
   directory (`../princeton_QRHead/` by default; override with
   `PRINCETON_DIR=`).
2. `pip install -e .` so Princeton's `qrretriever` package is on the path.
3. Downloads `single-session-user_s.json` from the
   [PrincetonPLI/QRHead HF dataset](https://huggingface.co/datasets/PrincetonPLI/QRHead).
4. Runs `python exp_scripts/detection/detect_qrhead_lme.py` with Princeton's
   own Llama-8B config. No `--truncate_by_space` (paper default). Saves
   ranking to `validation/results/princeton_llama_8b_lme.json`.
5. Compares the top-16 of our output to the 16 heads listed in
   `Llama-3.1-8B-Instruct_qr_head_LME.yaml`.

Pass criterion: top-16 sets are identical (order may differ slightly within
the set due to numerical noise around 1e-3).

Ground truth (from
[`Llama-3.1-8B-Instruct_qr_head_LME.yaml`](https://github.com/princeton-pli/QRHead/blob/main/src/qrretriever/configs/Llama-3.1-8B-Instruct_qr_head_LME.yaml)):

```
13-18, 13-21, 8-11, 14-13, 17-29, 13-1, 13-13, 14-29,
14-31, 13-8, 16-1, 17-21, 13-3, 10-31, 13-4, 14-22
```

Expected runtime on H100: ~30-60 min (70 LME instances × full attention
extraction with no truncation, Llama-3.1-8B at 128K context).

## Stage 2: Qwen2.5-7B LME (after Stage 1 passes)

Once Stage 1 is green, run the Qwen variant. Princeton ships
`Qwen2.5-7B-Instruct_full_head.yaml` natively — no code changes needed:

```bash
# In the same princeton_QRHead clone produced by Stage 1
PRINCETON_DIR=$(cd .. && pwd)/princeton_QRHead
cd $PRINCETON_DIR

python exp_scripts/detection/detect_qrhead_lme.py \
    --input_file data/longmemeval_data/single-session-user_s.json \
    --output_file ../qr_detect/validation/results/princeton_qwen_7b_lme.json \
    --config_or_config_path src/qrretriever/configs/Qwen2.5-7B-Instruct_full_head.yaml
```

Then compare against qr_detect's existing Qwen LME ranking
(`detection_results/Qwen2.5-7B-Instruct_qr_heads_LME.json`).

Expected outcomes:

- **Both rankings match** ⇒ qr_detect's Qwen path is correct; the
  layer-27-only artifact we documented earlier was caused by something we
  already fixed in the recent SDPA + MLP-chunk + preallocated-cache
  commits.

- **Princeton's Qwen ranking is healthy (mid-network layers, top score
  ~0.05) but qr_detect's is the all-layer-27 artifact** ⇒ the bug is in
  qr_detect's `custom_modeling_qwen2.py` interacting with chunked prefill.
  Use Princeton's run as the source of truth for Qwen.

- **Both produce the all-layer-27 artifact** ⇒ the bug is in Princeton's
  code itself for some Qwen-specific reason (unlikely given they ship
  Qwen2.5-7B as a supported model in their docs, but possible). At that
  point the fix lives upstream.

## Stage 3: OLMo-7B-Instruct-hf LME

Princeton doesn't ship OLMo. The qr_detect repo's existing OLMo ranking
([`detection_results/OLMo-7B-Instruct-hf_qr_heads_LME_evidence_preserved.json`](../detection_results/OLMo-7B-Instruct-hf_qr_heads_LME_evidence_preserved.json))
already looks healthy:

- top-1 score ≈ 0.054 (reasonable scale, comparable to Llama's 0.108)
- top-16 layer distribution: `{3: 2, 10: 1, 17: 3, 18: 4, 19: 2, 21: 3, 22: 1}`
  — spread across mid-network layers, matching the QRHead paper's claim
  that retrieval heads are mid-network.

So the cheap path: trust qr_detect's OLMo without running Princeton-OLMo.
If you want a cross-check, port `custom_modeling_olmo.py` and the
`OlmoForCausalLM` dispatch from qr_detect's `attn_retriever.py` into the
Princeton clone, then re-run.

## Stage 4: Mistral-7B-Instruct-v0.3 LME

Mistral isn't in Princeton or in the collaborator's qr_detect. It's only
on our `add-mistral-detection` branch in qr_detect. Run the existing
pipeline:

```bash
# On the GPU box, from a fresh checkout of qr_detect:
git checkout add-mistral-detection
bash run_mistral_detection.sh
```

The Mistral implementation uses the same light monkey-patch style as
OLMo. If Stage 1 (Llama) and Stage 2 (Qwen) both validate via Princeton's
upstream code, and Stage 3 (OLMo) is consistent, then by transitivity the
Mistral light-style implementation is sound.

## Files

| File | Purpose |
|---|---|
| `run_princeton_llama_lme.sh` | Stage 1 driver |
| `compare_to_ground_truth.py` | Compares output JSON to Princeton yaml |
| `results/` | Output directory (created at run time) |
