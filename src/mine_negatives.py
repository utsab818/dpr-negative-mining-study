"""Versioned, answer-filtered DPR hard-negative mining for V3.

Run from the repository root:
    python3 -m src.mine_negatives --num-questions 16 --num-passages 64 \
        --batch-size 4 --output-dir experiments/v3/mining/smoke

Full run (intended for Colab A100):
    python3 -m src.mine_negatives --num-questions 2000 --num-passages 5000 \
        --batch-size 32 --output-dir experiments/v3/mining/full
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.data import DPRDataset
from src.model import DPR, MODEL_NAME


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fingerprint_texts(texts):
    """Order-sensitive hash; separates items unambiguously by length."""
    digest = hashlib.sha256()
    for text in texts:
        value = text.encode("utf-8")
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


def encode_texts(model, tokenizer, texts, kind, device, batch_size, max_length):
    embeddings = []
    encoder = model.encode_passage if kind == "passage" else model.encode_question
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            inputs = tokenizer(
                chunk, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt"
            ).to(device)
            encoded = encoder(inputs["input_ids"], inputs["attention_mask"])
            embeddings.append(encoded.float().cpu())
            if start == 0 or (start // batch_size + 1) % 25 == 0:
                print(f"Encoded {kind}s: {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    return torch.cat(embeddings, dim=0)


def mine(Q, P, answers_by_query, passages, num_hard):
    """Scores on CPU; scans ranked candidates until K pass answer filtering."""
    scores = Q @ P.T
    n_questions, n_passages = scores.shape
    assert n_questions <= n_passages
    positive_ids = torch.arange(n_questions)

    # Diagnostics BEFORE excluding the designated positive.
    positive_scores = scores[positive_ids, positive_ids].clone()
    # Counts outranking passages; ties can make this differ from stable sort rank.
    positive_ranks = (scores > positive_scores[:, None]).sum(dim=1) + 1

    scores[positive_ids, positive_ids] = -float("inf")
    hardest_scores = scores.max(dim=1).values
    margins = positive_scores - hardest_scores

    hard_ids = torch.empty((n_questions, num_hard), dtype=torch.long)
    hard_scores = torch.empty((n_questions, num_hard), dtype=torch.float32)
    rejected_answer_counts = []
    scanned_counts = []
    folded_passages = [p.casefold() for p in passages]

    for i in range(n_questions):
        # Return sorted IDs for this question; one row at a time avoids
        # allocating another full [num_questions, num_passages] ranking matrix.
        ranking = torch.argsort(scores[i], descending=True)
        answers = [a.strip().casefold() for a in answers_by_query[i] if a.strip()]
        chosen = []
        rejected = 0
        examined = 0
        for j_tensor in ranking:
            j = int(j_tensor)
            if j == i:
                continue
            examined += 1
            if any(answer in folded_passages[j] for answer in answers):
                rejected += 1
                continue
            chosen.append(j)
            if len(chosen) == num_hard:
                break

        if len(chosen) < num_hard:
            raise RuntimeError(f"Only {len(chosen)} valid negatives for query {i}")
        hard_ids[i] = torch.tensor(chosen, dtype=torch.long)
        hard_scores[i] = scores[i, chosen]
        rejected_answer_counts.append(rejected)
        scanned_counts.append(examined)

    diagnostics = {
        "mean_positive_rank": float(positive_ranks.float().mean().item()),
        "median_positive_rank": float(positive_ranks.float().median().item()),
        "positive_rank_1_count": int((positive_ranks == 1).sum().item()),
        "negative_margin_count": int((margins < 0).sum().item()),
        "mean_positive_negative_margin": float(margins.mean().item()),
        "recall_at_1": float((positive_ranks <= 1).float().mean().item()),
        "recall_at_5": float((positive_ranks <= 5).float().mean().item()),
        "recall_at_20": float((positive_ranks <= 20).float().mean().item()),
        "mean_answer_rejections_per_query": sum(rejected_answer_counts) / n_questions,
        "total_answer_rejections": sum(rejected_answer_counts),
        "max_candidates_scanned_for_one_query": max(scanned_counts),
    }
    return hard_ids, hard_scores, diagnostics


def main():
    parser = argparse.ArgumentParser(description="V3 DPR hard-negative miner")
    parser.add_argument("--num-questions", type=int, default=2000)
    parser.add_argument("--num-passages", type=int, default=5000)
    parser.add_argument("--num-hard", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default="dpr_phase1.pt")
    parser.add_argument("--output-dir", default="experiments/v3/mining/full")
    args = parser.parse_args()

    if not (1 <= args.num_questions <= args.num_passages):
        parser.error("Require 1 <= num-questions <= num-passages")
    if not (1 <= args.num_hard < args.num_passages):
        parser.error("Require 1 <= num-hard < num-passages")
    if args.batch_size < 1:
        parser.error("batch-size must be positive")

    output_dir = Path(args.output_dir)
    # Never overwrite an existing completed run.
    for filename in ("mined_negatives.pt", "passage_index.pt", "manifest.json"):
        if (output_dir / filename).exists():
            raise FileExistsError(f"Output exists: {output_dir / filename}")
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"Warm-up checkpoint not found: {args.checkpoint}")

    torch.manual_seed(args.seed)
    device = choose_device()
    print("Device:", device, flush=True)
    print("Questions:", args.num_questions, "Passages:", args.num_passages, flush=True)

    # A single deterministic dataset prefix guarantees training positives
    # occupy passage IDs 0..num_questions-1 in the mining corpus.
    dataset = DPRDataset(split="train", max_samples=args.num_passages)
    if len(dataset.examples) != args.num_passages:
        raise RuntimeError("Dataset provided fewer unique passages than requested")
    passages = [e["passage"] for e in dataset.examples]
    subset = dataset.examples[:args.num_questions]
    questions = [e["question"] for e in subset]
    answers_by_query = [e["answers"] for e in subset]
    passage_hash = fingerprint_texts(passages)
    question_hash = fingerprint_texts(questions)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = DPR().to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    P = encode_texts(model, tokenizer, passages, "passage", device, args.batch_size, 192)
    Q = encode_texts(model, tokenizer, questions, "question", device, args.batch_size, 48)
    print("Passage embeddings:", tuple(P.shape))
    print("Question embeddings:", tuple(Q.shape))

    indices, values, diagnostics = mine(Q, P, answers_by_query, passages, args.num_hard)
    assert indices.shape == (args.num_questions, args.num_hard)
    assert all(i not in indices[i].tolist() for i in range(args.num_questions))
    print("Hard-negative indices:", tuple(indices.shape))
    print("Diagnostics:", json.dumps(diagnostics, indent=2))

    print("\nExample question:", questions[0])
    print("Positive passage:", passages[0][:200])
    for rank, j in enumerate(indices[0].tolist(), 1):
        print(f"Hard negative {rank} (passage {j}, score {values[0, rank-1]:.3f}):", passages[j][:200])

    manifest = {
        "version": "v3",
        "split": "train",
        "dataset": "squad",
        "selection": "first unique contexts; one question per passage",
        "num_questions": args.num_questions,
        "num_passages": args.num_passages,
        "num_hard": args.num_hard,
        "positive_passage_id_rule": "question_id_equals_passage_id",
        "passages_sha256": passage_hash,
        "questions_sha256": question_hash,
        "model_name": MODEL_NAME,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "question_max_length": 48,
        "passage_max_length": 192,
        "negative_filter": "exclude known positive and annotated-answer substring casefold",
        "diagnostics": diagnostics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"passages": passages, "embeddings": P,
                "passages_sha256": passage_hash}, output_dir / "passage_index.pt")
    torch.save({"questions": questions, "passages": passages,
                "hard_negative_indices": indices,
                "hard_negative_scores": values,
                "positive_passage_ids": torch.arange(args.num_questions),
                "questions_sha256": question_hash,
                "passages_sha256": passage_hash}, output_dir / "mined_negatives.pt")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Saved V3 mining artifacts to:", output_dir)


if __name__ == "__main__":
    main()
