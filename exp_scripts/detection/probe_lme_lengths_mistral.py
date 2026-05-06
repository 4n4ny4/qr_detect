"""
Probe: pick `--truncate_by_space N` for Mistral-7B-Instruct-v0.3 on
LongMemEval `single-session-user_s.json`.

Mistral has a 32K positional-encoding window; LME instances are roughly
50K-120K tokens at full length (Llama-3.1-8B sees them at 128K). We use
`--truncate_by_space N` to chop each paragraph (dialogue round) to its first
N words. The decision rule (see plan):

    Smallest N in [50, 100, 200, 300, 400] satisfying BOTH:
      (i)  fit32k(N) >= 60     # >= 60 of 70 instances fit Mistral's 32K window
      (ii) gold_intact(N) >= 0.90   # >= 90% of gold rounds fit untruncated
                                    # (so answer-bearing text isn't chopped)
    If no N works -> SKIP_MISTRAL_LME.

This script is CPU-only and does not load model weights; it only loads the
Mistral tokenizer (fast, downloaded once) to compute realistic token counts
that match what the detection script will see.

Outputs:
  - Prints a token-length table and gold-round word-count distribution.
  - Prints a single decision line: `chosen_N=...` or `SKIP_MISTRAL_LME`.
  - Writes the same content to results/probe_report.txt for citation in
    results/README.md.

Usage (from repo root):
  python exp_scripts/detection/probe_lme_lengths_mistral.py \\
      --input_file data/longmemeval_data/single-session-user_s.json \\
      --output_report results/probe_report.txt
"""

import argparse
import json
import os
import sys
from pathlib import Path

from transformers import AutoTokenizer

# --- Decision-rule constants. Mirror Step 6 of the plan. ---
N_CANDIDATES = [50, 100, 200, 300, 400]   # smallest first; rule picks smallest passing
CONTEXT_WINDOW = 32_000   # Mistral-7B-Instruct-v0.3 max position embeddings
FIT_THRESHOLD = 60        # >= 60 of 70 LME instances must fit
GOLD_THRESHOLD = 0.90     # >= 90% of gold rounds must fit untruncated

# The detection-script prompt template. Mirrors `attn_retriever.get_prompt`
# for OLMo / Mistral (chat-template path) so the token counts here match
# what `detect_qrhead_lme.py` will actually feed Mistral.
RETRIEVAL_INSTRUCTION = " Here are some paragraphs:"
RETRIEVAL_INSTRUCTION_LATE = (
    "Please find information that are relevant to the following query "
    "in the paragraphs above."
)
PROMPT_SEPARATOR = "\n\n"


def render_user_content(instance, N=None):
    """Reproduce the user-content body that `attn_retriever.get_prompt` builds.

    If N is not None, each paragraph_text is truncated to its first N
    whitespace-separated tokens before being inserted (matches what
    `--truncate_by_space N` does inside the detection scripts).
    """
    body = RETRIEVAL_INSTRUCTION
    for i, p in enumerate(instance["paragraphs"]):
        text = p["paragraph_text"].strip()
        if p.get("title"):
            text = p["title"] + "\n" + text
        if N is not None and N > 0:
            words = text.split()
            if len(words) > N:
                text = " ".join(words[:N])
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


def collect_gold_word_counts(data):
    """List of `len(paragraph_text.split())` for every gold round across all
    instances with non-empty `gt_docs`. Excludes abstention examples."""
    word_counts = []
    n_abstention = 0
    n_with_gold = 0
    for d in data:
        gold = set(d.get("gt_docs", []))
        if not gold:
            n_abstention += 1
            continue
        n_with_gold += 1
        for p in d["paragraphs"]:
            if p["idx"] in gold:
                word_counts.append(len(p["paragraph_text"].split()))
    word_counts.sort()
    return word_counts, n_abstention, n_with_gold


def fit32k_at(tokenizer, data, N):
    """Count instances whose tokenized prompt <= CONTEXT_WINDOW after
    truncating each paragraph to first N words."""
    fit = 0
    lengths = []
    for d in data:
        body = render_user_content(d, N=N)
        prompt = render_chat_prompt(tokenizer, body)
        n_tok = len(tokenizer.encode(prompt))
        lengths.append(n_tok)
        if n_tok <= CONTEXT_WINDOW:
            fit += 1
    lengths.sort()
    return fit, lengths


def gold_intact_at(gold_word_counts, N):
    if not gold_word_counts:
        return 0.0
    return sum(1 for c in gold_word_counts if c <= N) / len(gold_word_counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/longmemeval_data/single-session-user_s.json",
        help="LME single-session-user detection JSON (70 instances).",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="HF tokenizer id. Loads tokenizer only; no model weights.",
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default="results/probe_report.txt",
        help="Where to write the textual probe report.",
    )
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.tokenizer}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"Reading LME data: {args.input_file}", flush=True)
    with open(args.input_file, "r") as f:
        data = json.load(f)
    print(f"  -> {len(data)} instances", flush=True)

    # --- Token-length distribution at each N ---
    table_lines = []
    table_lines.append(
        f"\nToken-length distribution (Mistral tokenizer, chat template, CONTEXT_WINDOW={CONTEXT_WINDOW}):"
    )
    table_lines.append(f"{'N':<8}| {'min':<7} {'median':<8} {'max':<7} {'fit32k/' + str(len(data))}")
    table_lines.append("-" * 50)

    fit32k_by_N = {}
    for N in [None] + N_CANDIDATES:
        fit, lengths = fit32k_at(tokenizer, data, N)
        if N is None:
            fit32k_by_N[None] = fit
            label = "full"
        else:
            fit32k_by_N[N] = fit
            label = f"N={N}"
        median = lengths[len(lengths) // 2]
        table_lines.append(
            f"{label:<8}| {min(lengths):<7} {median:<8} {max(lengths):<7} {fit}/{len(data)}"
        )

    # --- Gold-round word-count distribution ---
    gold_word_counts, n_abstention, n_with_gold = collect_gold_word_counts(data)
    table_lines.append("")
    table_lines.append(
        f"Gold rounds: {len(gold_word_counts)} rounds across {n_with_gold} instances "
        f"(plus {n_abstention} abstention instances with empty gt_docs, excluded)."
    )
    if gold_word_counts:
        q1 = gold_word_counts[len(gold_word_counts) // 4]
        q2 = gold_word_counts[len(gold_word_counts) // 2]
        q3 = gold_word_counts[3 * len(gold_word_counts) // 4]
        q_max = gold_word_counts[-1]
        table_lines.append(
            f"  word-count quantiles: p25={q1} p50={q2} p75={q3} max={q_max}"
        )
        table_lines.append("  fit-fully-at-N (gold round word_count <= N):")
        gold_intact_by_N = {}
        for N in N_CANDIDATES:
            pct = 100 * gold_intact_at(gold_word_counts, N)
            gold_intact_by_N[N] = pct / 100.0
            table_lines.append(f"    N={N}: {pct:.0f}%")
    else:
        gold_intact_by_N = {N: 0.0 for N in N_CANDIDATES}
        table_lines.append("  (no gold rounds present in data)")

    # --- Decision rule ---
    table_lines.append("")
    table_lines.append(
        f"Decision rule: smallest N s.t. fit32k(N) >= {FIT_THRESHOLD} "
        f"AND gold_intact(N) >= {GOLD_THRESHOLD:.2f}"
    )
    chosen_N = None
    for N in N_CANDIDATES:
        if fit32k_by_N.get(N, 0) >= FIT_THRESHOLD and gold_intact_by_N.get(N, 0.0) >= GOLD_THRESHOLD:
            chosen_N = N
            break

    if chosen_N is None:
        table_lines.append("DECISION: SKIP_MISTRAL_LME")
        table_lines.append(
            "  No N in [50, 100, 200, 300, 400] satisfies both conditions. "
            "Mistral cannot run on LME without methodologically-questionable truncation. "
            "Deliver only results/mistral_nq.json."
        )
    else:
        table_lines.append(f"DECISION: chosen_N={chosen_N}")
        table_lines.append(
            f"  Run: python exp_scripts/detection/detect_qrhead_lme.py "
            f"--input_file {args.input_file} "
            f"--output_file results/mistral_lme.json "
            f"--truncate_by_space {chosen_N} "
            f"--config_or_config_path src/qrretriever/configs/"
            f"Mistral-7B-Instruct-v0.3_full_head.yaml"
        )

    # --- Emit ---
    report = "\n".join(table_lines)
    print(report, flush=True)

    out_path = Path(args.output_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"Probe report (tokenizer={args.tokenizer}, input={args.input_file})\n")
        f.write(report + "\n")
    print(f"\nWrote probe report -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
