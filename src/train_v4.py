"""V3 DPR second-stage trainer.

Run as: python3 -m src.train_v3 --strategy hard --mined-path ... --warmup-path ...

All strategies may use the same question batch size. For inbatch, the number
of negatives per query is implicitly batch_size - 1.
"""

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer

from src.data import DPRDataset
from src.loss import contrastive_loss, hard_negative_loss, mixed_negative_loss
from src.model import DPR, MODEL_NAME


def device_for_training():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize(tokenizer, texts, max_length, device):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)


def batch_id_generator(num_examples, batch_size, seed):
    """Shuffle each epoch, dropping only the final incomplete batch."""
    if not 1 <= batch_size <= num_examples:
        raise ValueError("Batch size must be between 1 and dataset size")
    generator = torch.Generator().manual_seed(seed)
    while True:
        order = torch.randperm(num_examples, generator=generator).tolist()
        for start in range(0, num_examples - batch_size + 1, batch_size):
            yield order[start:start + batch_size]


def fingerprint_texts(texts):
    hasher = hashlib.sha256()
    for value in texts:
        encoded = value.encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, "big"))
        hasher.update(encoded)
    return hasher.hexdigest()


def train(args):
    set_seed(args.seed)
    device = device_for_training()
    print("Device:", device, flush=True)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Strategy:", args.strategy, "Seed:", args.seed, flush=True)

    mined = torch.load(args.mined_path, map_location="cpu", weights_only=True)
    questions = mined["questions"]
    passages = mined["passages"]
    hard_ids = mined["hard_negative_indices"]
    n = len(questions)
    if not 1 <= n <= len(passages):
        raise ValueError("Invalid questions/passages sizes")
    if hard_ids.ndim != 2 or hard_ids.size(0) != n:
        raise ValueError("Hard negative rows must match questions")
    if args.batch_size > n:
        raise ValueError("Batch size exceeds the number of training questions")
    if args.strategy in ("hard", "mixed") and args.num_negatives > hard_ids.size(1):
        raise ValueError("Requested K exceeds the number of mined hard negatives")
    if args.strategy in ("hard", "mixed"):
        chosen = hard_ids[:, :args.num_negatives]
        if (chosen < 0).any() or (chosen >= len(passages)).any():
            raise ValueError("Hard negative index outside the passage collection")
        if (chosen == torch.arange(n).unsqueeze(1)).any():
            raise ValueError("Hard negatives contain a known positive ID")

    # The first n passages are the annotated positives by construction.
    dataset = DPRDataset(split="train", max_samples=n)
    if len(dataset.examples) != n or any(
        dataset.examples[i]["question"] != questions[i]
        or dataset.examples[i]["passage"] != passages[i]
        for i in range(n)
    ):
        raise ValueError("Mining artifact does not align with SQuAD training examples")

    # V4: avoid materializing O(num_questions * num_passages) random-candidate lists.
    # Rejection-sample eligible passages on demand (unbiased over eligible IDs).
    folded_passages = [p.casefold() for p in passages] if args.strategy == "random" else None
    folded_answers = (
        [[a.strip().casefold() for a in example["answers"] if a.strip()]
         for example in dataset.examples]
        if args.strategy == "random" else None
    )
    if args.strategy == "random" and len(passages) - 1 < args.num_negatives:
        raise ValueError("Too few candidate passages")

    target_steps = args.max_steps
    if target_steps is None:
        target_steps = n // args.batch_size
    if target_steps < 1:
        raise ValueError("No optimizer steps requested")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"{args.strategy}_k{args.num_negatives if args.strategy != 'inbatch' else args.batch_size - 1}"
        f"_b{args.batch_size}_seed{args.seed}_steps{target_steps}"
    )
    checkpoint_path = output_dir / ("dpr_" + tag + ".pt")
    metrics_path = output_dir / ("dpr_" + tag + ".json")
    steps_path = output_dir / ("dpr_" + tag + "_steps.csv")
    for path in (checkpoint_path, metrics_path, steps_path):
        if path.exists():
            raise FileExistsError("Refusing to overwrite existing V3 result: %s" % path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = DPR()
    state = torch.load(args.warmup_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    if args.gradient_checkpointing:
        model.question_encoder.gradient_checkpointing_enable()
        model.passage_encoder.gradient_checkpointing_enable()
    model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    batches = batch_id_generator(n, args.batch_size, args.seed)
    negative_rng = random.Random(args.seed)

    total_loss = 0.0
    max_memory = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Synchronize to measure accelerator work rather than CPU launch time.
    synchronize(device)
    t0 = time.perf_counter()
    question_encodings = 0
    passage_encodings = 0

    with steps_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "loss", "seconds_elapsed"])
        writer.writeheader()
        for step in range(1, target_steps + 1):
            batch_ids = next(batches)
            q_inputs = tokenize(tokenizer, [questions[i] for i in batch_ids], 48, device)
            p_inputs = tokenize(tokenizer, [passages[i] for i in batch_ids], 192, device)
            neg_inputs = None
            if args.strategy != "inbatch":
                neg_ids = []
                for i in batch_ids:
                    if args.strategy == "random":
                        selected = []
                        selected_set = set()
                        attempts = 0
                        while len(selected) < args.num_negatives:
                            attempts += 1
                            if attempts > 100000:
                                raise RuntimeError(f"No eligible random negatives for query {i}")
                            j = negative_rng.randrange(len(passages))
                            if j == i or j in selected_set:
                                continue
                            if any(a in folded_passages[j] for a in folded_answers[i]):
                                continue
                            selected.append(j)
                            selected_set.add(j)
                    else:
                        selected = hard_ids[i, :args.num_negatives].tolist()
                    neg_ids.extend(selected)
                neg_inputs = tokenize(tokenizer, [passages[j] for j in neg_ids], 192, device)

            optimizer.zero_grad()
            q = model.encode_question(q_inputs["input_ids"], q_inputs["attention_mask"])
            p_pos = model.encode_passage(p_inputs["input_ids"], p_inputs["attention_mask"])
            if args.strategy == "inbatch":
                loss, scores = contrastive_loss(q, p_pos)
            else:
                p_neg = model.encode_passage(
                    neg_inputs["input_ids"], neg_inputs["attention_mask"]
                ).reshape(len(batch_ids), args.num_negatives, -1)
                if args.strategy == "mixed":
                    loss, scores = mixed_negative_loss(q, p_pos, p_neg)
                else:
                    loss, scores = hard_negative_loss(q, p_pos, p_neg)
            loss.backward()
            optimizer.step()
            synchronize(device)
            elapsed = time.perf_counter() - t0
            value = loss.item()
            total_loss += value
            question_encodings += len(batch_ids)
            passage_encodings += len(batch_ids) * (1 if args.strategy == "inbatch" else 1 + args.num_negatives)
            writer.writerow({"step": step, "loss": value, "seconds_elapsed": round(elapsed, 3)})
            handle.flush()
            if step == 1 or step % args.log_every == 0 or step == target_steps:
                print(
                    f"Step {step} | Scores {tuple(scores.shape)} | "
                    f"Loss {value:.4f} | Elapsed {elapsed:.1f}s",
                    flush=True,
                )

    elapsed = time.perf_counter() - t0
    if device.type == "cuda":
        max_memory = round(torch.cuda.max_memory_allocated() / (1024**3), 3)

    metrics = {
        "version": "v4",
        "strategy": args.strategy,
        "seed": args.seed,
        "num_questions": n,
        "num_passages": len(passages),
        "batch_size": args.batch_size,
        "explicit_negatives_per_query": 0 if args.strategy == "inbatch" else args.num_negatives,
        "inbatch_negatives_per_query": args.batch_size - 1 if args.strategy in ("mixed", "inbatch") else 0,
        "optimizer_steps": target_steps,
        "query_exposures": question_encodings,
        "passage_encoder_input_count": passage_encodings,
        "mean_train_loss": total_loss / target_steps,
        "train_seconds": round(elapsed, 3),
        "cuda_peak_allocated_gb": max_memory,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "gradient_checkpointing": args.gradient_checkpointing,
        "learning_rate": args.lr,
        "mined_path": str(args.mined_path),
        "warmup_path": str(args.warmup_path),
        "question_sha256": fingerprint_texts(questions),
        "passage_sha256": fingerprint_texts(passages),
        "sampler": "seeded_epoch_permutation_drop_last",
        "note": "In-batch positives may be false negatives for other queries; matching query batch size does not match FLOPs.",
    }
    torch.save(model.state_dict(), checkpoint_path)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print("Total optimizer steps:", target_steps)
    print("Query exposures:", question_encodings)
    print("Passage encoder inputs:", passage_encodings)
    print("Average loss:", metrics["mean_train_loss"])
    print("Training seconds:", elapsed)
    print("CUDA peak allocated GB:", max_memory)
    print("Saved:", checkpoint_path)
    print("Metrics:", metrics_path)
    print("Per-step log:", steps_path)


def main():
    parser = argparse.ArgumentParser(description="V4 DPR training with lazy random-negative sampling")
    parser.add_argument("--strategy", choices=["random", "hard", "mixed", "inbatch"], required=True)
    parser.add_argument("--mined-path", type=Path, default=Path("experiments/v3/mining/full/mined_negatives.pt"))
    parser.add_argument("--warmup-path", type=Path, default=Path("experiments/v2/inputs/dpr_phase1.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/v3/training"))
    parser.add_argument("--num-negatives", type=int, default=4, help="Explicit negatives per query; ignored for inbatch")
    parser.add_argument("--batch-size", type=int, default=33)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Reduce activation memory at extra compute cost")
    args = parser.parse_args()
    if args.batch_size < 2:
        parser.error("--batch-size must be at least 2 for comparable training")
    if args.num_negatives < 1:
        parser.error("--num-negatives must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.log_every < 1:
        parser.error("--log-every must be positive")
    train(args)


if __name__ == "__main__":
    main()
