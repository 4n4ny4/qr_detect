"""
Probe: pick `--truncate_by_space N` for Mistral-7B-Instruct-v0.3 on
LongMemEval `single-session-user_s.json`, using the evidence-preserving
truncation introduced for OLMo (commit 8d9686e).

Background. Mistral has a 32K positional-encoding window; LME instances
are roughly 50K-120K tokens at full length (Llama-3.1-8B sees them at
128K). The collaborator's `--evidence_preserving_truncation` flag on
`detect_qrhead_lme.py` truncates ONLY non-gold paragraphs to
`--truncate_by_space N`, keeping gold rounds full (or capped via
`--gold_truncate_by_space`, default 0 = preserve fully). Because gold
rounds are preserved by construction, the answer-bearing text is never
chopped, so the QRScore signal stays clean.

Decision rule (simplified vs. uniform truncation):

    Smallest N in [5, 25, 50, 100, 200, 400] such that
      fit32k_ev(N) >= 60   # >= 60 of 70 instances fit Mistral's 32K
                           # window when only non-gold paragraphs are
                           # truncated to N words.
    If no N works -> SKIP_MISTRAL_LME.

For comparison, the script also reports the OLD uniform-truncation
fit32k at each N (no evidence preservation, all paragraphs truncated).

CPU-only; loads only the Mistral tokenizer (no model weights).

Usage (from repo root):
  python exp_scripts/detection/probe_lme_lengths_mistral.py \\
      --input_file data/longmemeval_data/single-session-user_s.json \\
      --output_report results/probe_report.txt
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

# --- Decision-rule constants. Mirror plan Step 6 (evidence-preserving variant). ---
N_CANDIDATES = [5, 25, 50, 100, 200, 400]   # smallest first; rule picks smallest passing
CONTEXT_WINDOW = 32_000   # Mistral-7B-Instruct-v0.3 max position embeddings
FIT_THRESHOLD = 60        # >= 60 of 70 LME instances must fit

# Detection-script prompt template. Mirrors `attn_retriever.get_prompt` for
# OLMo / Mistral (chat-template path) so the token counts here match what
# `detect_qrhead_lme.py` will actually feed Mistral at run time.
RETRIEVAL_INSTRUCTION = " Here are some paragraphs:"
RETRIEVAL_INSTRUCTION_LATE = (
    "Please find information that are relevant to the following query "
    "in the paragraphs above."
)
PROMPT_SEPARATOR = "\n\n"


def render_user_content(instance, distractor_N=None, gold_N=None):
    """Reproduce the user-content body that `attn_retriever.get_prompt` builds.

    distractor_N: word cap on non-gold paragraphs (None = no truncation)
    gold_N: word cap on gold paragraphs (None or 0 = preserve fully)
    """
    gold_idx = set(instance.get("gt_docs", []))
    body = RETRIEVAL_INSTRUCTION
    for i, p in enumerate(instance["paragraphs"]):
        text = p["paragraph_text"].strip()
        if p.get("title"):
            text = p["title"] + "\n" + text

        is_gold = (p.get("idx") in gold_idx) or (p.get("is_supporting") is True)
        if is_gold:
            cap = gold_N
        else:
            cap = distractor_N

        if cap is not None and cap > 0:
            words = text.split()
            if len(words) > cap:
                text = " ".join(words[:cap])

        body += PROMPT_SEPARATOR + f"[{i + 1}] {text}"
    body += PROMPT_SEPARATOR + RETRIEVAL_INSTRUCTION_LATE + PROMPT_SEPARATOR + "Query:"
    body += " " + instance["question"]
    return body


def render_chat_prompt(tokenizer, user_content):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def measure_lengths(tokenizer, data, distractor_N, gold_N):
    """Return (fit_count, sorted_lengths) for a given distractor/gold cap pair."""
    lengths = []
    for d in data:
        body = render_user_content(d, distractor_N=distractor_N, gold_N=gold_N)
        prompt = render_chat_prompt(tokenizer, body)
        lengths.append(len(tokenizer.encode(prompt)))
    lengths.sort()
    fit = sum(1 for l in lengths if l <= CONTEXT_WINDOW)
    return fit, lengths


def _q(sorted_lengths, q):
    if not sorted_lengths:
        return 0
    idx = max(0, min(len(sorted_lengths) - 1, int(q * len(sorted_lengths))))
    return sorted_lengths[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/longmemeval_data/single-session-user_s.json",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default="results/probe_report.txt",
    )
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.tokenizer}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"Reading LME data: {args.input_file}", flush=True)
    with open(args.input_file, "r") as f:
        data = json.load(f)
    print(f"  -> {len(data)} instances", flush=True)

    n_with_gold = sum(1 for d in data if d.get("gt_docs"))
    n_abstention = len(data) - n_with_gold

    lines = []
    lines.append(
        f"Probe (tokenizer={args.tokenizer}, input={args.input_file}, "
        f"CONTEXT_WINDOW={CONTEXT_WINDOW})"
    )
    lines.append(
        f"Total instances: {len(data)}  with-gold: {n_with_gold}  "
        f"abstention (empty gt_docs): {n_abstention}"
    )

    # --- Baseline: full prompt (no truncation) ---
    full_fit, full_lens = measure_lengths(tokenizer, data, distractor_N=None, gold_N=None)
    lines.append("")
    lines.append(
        f"Full prompt (no truncation): "
        f"min={min(full_lens)} median={_q(full_lens, 0.5)} max={max(full_lens)} "
        f"fit32k={full_fit}/{len(data)}"
    )

    # --- Evidence-preserving truncation (RECOMMENDED): gold full, distractors -> N ---
    lines.append("")
    lines.append("=== Evidence-preserving truncation (gold rounds full, distractors -> N) ===")
    lines.append(f"{'N':<6}| {'min':<6} {'median':<7} {'max':<7} {'fit32k/' + str(len(data))}")
    lines.append("-" * 50)
    fit_ev = {}
    for N in N_CANDIDATES:
        fit, lens = measure_lengths(tokenizer, data, distractor_N=N, gold_N=None)
        fit_ev[N] = fit
        lines.append(
            f"N={N:<4}| {min(lens):<6} {_q(lens, 0.5):<7} {max(lens):<7} {fit}/{len(data)}"
        )

    # --- Comparison: uniform truncation (gold + distractors both -> N) ---
    lines.append("")
    lines.append("=== Comparison: UNIFORM truncation (gold AND distractors -> N) ===")
    lines.append("(NOT recommended: chops gold answer text. Reference only.)")
    lines.append(f"{'N':<6}| {'min':<6} {'median':<7} {'max':<7} {'fit32k/' + str(len(data))}")
    lines.append("-" * 50)
    fit_uniform = {}
    for N in N_CANDIDATES:
        fit, lens = measure_lengths(tokenizer, data, distractor_N=N, gold_N=N)
        fit_uniform[N] = fit
        lines.append(
            f"N={N:<4}| {min(lens):<6} {_q(lens, 0.5):<7} {max(lens):<7} {fit}/{len(data)}"
        )

    # --- Decision rule (evidence-preserving) ---
    lines.append("")
    lines.append(
        f"Decision rule: smallest N in {N_CANDIDATES} s.t. "
        f"evidence-preserving fit32k(N) >= {FIT_THRESHOLD}"
    )
    chosen_N = None
    for N in N_CANDIDATES:
        if fit_ev.get(N, 0) >= FIT_THRESHOLD:
            chosen_N = N
            break

    if chosen_N is None:
        lines.append("DECISION: SKIP_MISTRAL_LME")
        lines.append(
            "  No N in [5, 25, 50, 100, 200, 400] yields >= 60/70 instances "
            "fitting Mistral's 32K window even with gold rounds preserved. "
            "Gold rounds plus per-instance prompt overhead exceed the window. "
            "Deliver only results/mistral_nq.json."
        )
    else:
        lines.append(f"DECISION: chosen_N={chosen_N}")
        lines.append(
            f"  Run: python exp_scripts/detection/detect_qrhead_lme.py "
            f"--input_file {args.input_file} "
            f"--output_file results/mistral_lme.json "
            f"--truncate_by_space {chosen_N} "
            f"--evidence_preserving_truncation "
            f"--config_or_config_path src/qrretriever/configs/"
            f"Mistral-7B-Instruct-v0.3_full_head.yaml"
        )

    report = "\n".join(lines)
    print("\n" + report, flush=True)

    out_path = Path(args.output_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report + "\n")
    print(f"\nWrote probe report -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
