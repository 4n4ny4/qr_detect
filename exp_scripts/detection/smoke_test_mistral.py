"""
2-instance / 3-paragraph smoke test for the Mistral detection wiring.

Confirms (in roughly 2 minutes on H100):
  * `custom_modeling_mistral.py` imports without ImportError.
  * `MistralForCausalLM.from_pretrained` succeeds with the SDPA / eager dispatch.
  * The monkey-patched `self_attn.forward` runs without signature mismatch.
  * The monkey-patched `mlp.forward` runs without errors.
  * `score_docs_per_head_for_detection` returns per-doc tensors of shape
    `(num_layers=32, num_heads=32)`.

Usage (from repo root):
  python exp_scripts/detection/smoke_test_mistral.py \\
      --input_file data/longmemeval_data/single-session-user_s.json
"""

import argparse
import json
import sys

from qrretriever.attn_retriever import FullHeadRetriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/longmemeval_data/single-session-user_s.json",
    )
    parser.add_argument(
        "--config_or_config_path",
        type=str,
        default="src/qrretriever/configs/Mistral-7B-Instruct-v0.3_full_head.yaml",
    )
    parser.add_argument("--num_examples", type=int, default=2)
    parser.add_argument("--num_paragraphs", type=int, default=3)
    args = parser.parse_args()

    print(f"Loading 2 LME instances from {args.input_file}", flush=True)
    with open(args.input_file, "r") as f:
        data = json.load(f)[: args.num_examples]

    print("Instantiating FullHeadRetriever (this will download Mistral weights if not cached)", flush=True)
    retriever = FullHeadRetriever(config_or_config_path=args.config_or_config_path)

    expected_layers = retriever.llm.config.num_hidden_layers
    expected_heads = retriever.llm.config.num_attention_heads
    print(
        f"Model loaded. Expecting per-doc tensors of shape ({expected_layers}, {expected_heads}).",
        flush=True,
    )

    for i, instance in enumerate(data):
        docs = instance["paragraphs"][: args.num_paragraphs]
        print(
            f"\nExample {i}: question={instance['question'][:80]!r} "
            f"num_docs={len(docs)} gt_docs={instance.get('gt_docs')}",
            flush=True,
        )
        try:
            scores = retriever.score_docs_per_head_for_detection(instance["question"], docs)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
            raise

        for doc_id, tensor in scores.items():
            shape = tuple(tensor.shape)
            ok = shape == (expected_layers, expected_heads)
            tag = "OK" if ok else "WRONG SHAPE"
            print(f"  doc {doc_id}: shape={shape} [{tag}]", flush=True)
            if not ok:
                sys.exit(2)

    print("\nSmoke test passed.", flush=True)


if __name__ == "__main__":
    main()
