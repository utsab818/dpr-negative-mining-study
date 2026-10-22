"""Evaluate DPR checkpoints on a fixed SQuAD retrieval benchmark.

Example:
python3 -m src.evaluate_v3 \
  --checkpoint warmup=experiments/v2/inputs/dpr_phase1.pt \
  --checkpoint hard_smoke=experiments/v3/training/smoke/dpr_hard_k4_b4_seed42_steps2.pt \
  --max-samples 1000 --batch-size 8 \
  --output experiments/v3/evaluation/smoke_1000.csv
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.data import DPRDataset
from src.model import DPR, MODEL_NAME


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fingerprint(texts):
    digest = hashlib.sha256()
    for text in texts:
        data = text.encode("utf-8")
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def encode_texts(texts, tokenizer, encode_fn, max_length, batch_size, device):
    output = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            inputs = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            embeddings = encode_fn(inputs["input_ids"], inputs["attention_mask"])
            output.append(embeddings.detach().float().cpu())
    return torch.cat(output, dim=0)


def evaluate_one(model, tokenizer, questions, passages, device, batch_size):
    q = encode_texts(questions, tokenizer, model.encode_question, 48, batch_size, device)
    p = encode_texts(passages, tokenizer, model.encode_passage, 192, batch_size, device)
    scores = q @ p.T
    ranks = torch.argsort(scores, dim=1, descending=True)
    positives = torch.arange(len(questions))
    # Current DPRDataset keeps one question per passage, indexed identically.
    if len(questions) != len(passages):
        raise ValueError("This benchmark requires one positive passage per question")
    matches = ranks.eq(positives.unsqueeze(1))
    positive_ranks = matches.to(torch.int64).argmax(dim=1) + 1
    metrics = {
        "recall@1": (positive_ranks <= 1).float().mean().item(),
        "recall@5": (positive_ranks <= 5).float().mean().item(),
        "recall@20": (positive_ranks <= 20).float().mean().item(),
        "mrr@10": torch.where(
            positive_ranks <= 10,
            1.0 / positive_ranks.float(),
            torch.zeros_like(positive_ranks, dtype=torch.float32),
        ).mean().item(),
    }
    return metrics, positive_ranks.tolist(), tuple(scores.shape)


def parse_checkpoints(items):
    parsed = []
    labels = set()
    for item in items:
        if "=" not in item:
            raise ValueError("Use --checkpoint label=path/to/weights.pt")
        label, path = item.split("=", 1)
        if not label or not path or label in labels:
            raise ValueError("Checkpoint labels must be nonempty and unique")
        labels.add(label)
        candidate = Path(path)
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        parsed.append((label, candidate))
    return parsed


def main():
    parser = argparse.ArgumentParser(description="Evaluate DPR checkpoints")
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="Repeatable argument: label=checkpoint.pt")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-query-ranks", action="store_true",
                        help="Write one positive-passage rank per query to a companion CSV")
    args = parser.parse_args()
    if args.max_samples < 20 or args.batch_size < 1:
        parser.error("--max-samples must be >=20 and --batch-size must be >=1")
    if args.output.exists():
        parser.error("Output already exists; select a new filename")
    rank_path = args.output.with_name(args.output.stem + "_query_ranks.csv")
    meta_path = args.output.with_name(args.output.stem + "_metadata.json")
    if args.save_query_ranks and rank_path.exists():
        parser.error("Query-rank output already exists")
    if meta_path.exists():
        parser.error("Metadata output already exists")

    checkpoints = parse_checkpoints(args.checkpoint)
    device = select_device()
    print("Device:", device)
    dataset = DPRDataset(split="validation", max_samples=args.max_samples)
    questions = [e["question"] for e in dataset.examples]
    passages = [e["passage"] for e in dataset.examples]
    print("Validation pairs:", len(questions))
    if len(questions) != args.max_samples:
        raise ValueError(f"Expected {args.max_samples} validation pairs; got {len(questions)}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    all_results = []
    per_query = []
    for label, checkpoint in checkpoints:
        print("\n=====", label.upper(), "=====")
        model = DPR().to(device)
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(weights)
        model.eval()
        metrics, positive_ranks, shape = evaluate_one(
            model, tokenizer, questions, passages, device, args.batch_size
        )
        print("Similarity matrix shape:", shape)
        for k, value in metrics.items():
            print(f"{k}: {value:.4f}")
        all_results.append({"strategy": label, "checkpoint": str(checkpoint), **metrics})
        if args.save_query_ranks:
            per_query.extend({"strategy": label, "query_id": i, "positive_rank": rank}
                             for i, rank in enumerate(positive_ranks))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["strategy", "checkpoint", "recall@1", "recall@5", "recall@20", "mrr@10"])
        writer.writeheader()
        writer.writerows(all_results)
    if args.save_query_ranks:
        with rank_path.open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["strategy", "query_id", "positive_rank"])
            writer.writeheader()
            writer.writerows(per_query)
    metadata = {
        "split": "validation", "num_pairs": len(questions),
        "question_sha256": fingerprint(questions),
        "passage_sha256": fingerprint(passages),
        "model_name": MODEL_NAME,
        "question_max_length": 48, "passage_max_length": 192,
        "device": str(device), "batch_size": args.batch_size,
        "positive_definition": "paired context at the same dataset index",
        "note": "Single annotated positive per query; other relevant passages may be unannotated",
    }
    with meta_path.open("x") as handle:
        json.dump(metadata, handle, indent=2)
    print("\nResults saved to", args.output)
    if args.save_query_ranks:
        print("Per-query ranks saved to", rank_path)
    print("Metadata saved to", meta_path)


if __name__ == "__main__":
    main()
