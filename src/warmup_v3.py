"""V3 DPR warm-up: in-batch contrastive training over unique SQuAD contexts.

Run from the repository root:
    python3 -m src.warmup_v3 --max-samples 16 --batch-size 4 --max-steps 2 \
        --output-dir experiments/v3/warmup/smoke

The resulting .pt checkpoint is a DPR state_dict compatible with the V3 miner
and trainer. Model weights start from the architecture's pretrained encoders
unless --init-checkpoint is explicitly supplied.
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
from src.loss import contrastive_loss
from src.model import DPR, MODEL_NAME


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def batch_id_generator(count, batch_size, seed):
    """Seeded epoch shuffling; incomplete last batch is dropped each epoch."""
    rng = torch.Generator().manual_seed(seed)
    while True:
        permutation = torch.randperm(count, generator=rng).tolist()
        for start in range(0, count - batch_size + 1, batch_size):
            yield permutation[start:start + batch_size]


def fingerprint(examples):
    digest = hashlib.sha256()
    for example in examples:
        for field in ("question", "passage"):
            value = example[field].encode("utf-8")
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="V3 DPR warm-up training")
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1,
                        help="Used when --max-steps is not specified")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("experiments/v3/warmup/full"))
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help="Optional DPR checkpoint; by default start from pretrained encoders")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    if args.max_samples < 2 or args.batch_size < 2 or args.batch_size > args.max_samples:
        parser.error("Require 2 <= batch-size <= max-samples")
    if args.epochs < 1 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("Require positive epochs and max-steps")
    if args.lr <= 0:
        parser.error("Learning rate must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = choose_device()
    print("Device:", device, flush=True)
    dataset = DPRDataset(split="train", max_samples=args.max_samples)
    examples = dataset.examples
    count = len(examples)
    if count < args.batch_size:
        raise ValueError("Not enough unique training examples for the requested batch size")
    steps_per_epoch = count // args.batch_size
    steps = args.max_steps if args.max_steps is not None else steps_per_epoch * args.epochs

    prefix = "dpr_warmup_n{}_b{}_seed{}_steps{}".format(
        count, args.batch_size, args.seed, steps)
    output_pt = args.output_dir / (prefix + ".pt")
    output_json = args.output_dir / (prefix + ".json")
    output_csv = args.output_dir / (prefix + "_steps.csv")
    for path in (output_pt, output_json, output_csv):
        if path.exists():
            raise FileExistsError("Existing experiment artifact: {}".format(path))

    print("Unique question/passage pairs:", count)
    print("Steps per full epoch:", steps_per_epoch)
    print("Optimizer steps:", steps)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = DPR()
    if args.init_checkpoint is not None:
        state = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print("Starting checkpoint:", args.init_checkpoint)
    else:
        print("Starting checkpoint: pretrained encoders (fresh DPR warm-up)")
    if args.gradient_checkpointing:
        model.question_encoder.gradient_checkpointing_enable()
        model.passage_encoder.gradient_checkpointing_enable()
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    sampler = batch_id_generator(count, args.batch_size, args.seed)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    logs = []
    loss_sum = 0.0
    synchronize(device)
    start_time = time.perf_counter()

    for step in range(1, steps + 1):
        ids = next(sampler)
        questions = [examples[i]["question"] for i in ids]
        passages = [examples[i]["passage"] for i in ids]
        q_inputs = tokenizer(questions, padding=True, truncation=True,
                             max_length=48, return_tensors="pt").to(device)
        p_inputs = tokenizer(passages, padding=True, truncation=True,
                             max_length=192, return_tensors="pt").to(device)

        optimizer.zero_grad(set_to_none=True)
        q = model.encode_question(q_inputs["input_ids"], q_inputs["attention_mask"])
        p = model.encode_passage(p_inputs["input_ids"], p_inputs["attention_mask"])
        loss, scores = contrastive_loss(q, p)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        loss_sum += loss_value
        if step == 1 or step % 10 == 0 or step == steps:
            synchronize(device)
            elapsed = time.perf_counter() - start_time
            print("Step {} | Scores {} | Loss {:.4f} | Elapsed {:.1f}s".format(
                step, tuple(scores.shape), loss_value, elapsed), flush=True)
        logs.append({"step": step, "loss": loss_value,
                     "query_exposures": step * args.batch_size})

    synchronize(device)
    elapsed = time.perf_counter() - start_time
    peak_gb = (torch.cuda.max_memory_allocated() / (1024 ** 3)
               if device.type == "cuda" else None)
    metadata = {
        "stage": "v3_warmup", "dataset": "squad", "split": "train",
        "num_examples": count, "dataset_fingerprint": fingerprint(examples),
        "encoder": MODEL_NAME, "seed": args.seed, "batch_size": args.batch_size,
        "optimizer_steps": steps, "query_exposures": steps * args.batch_size,
        "passage_encoder_inputs": steps * args.batch_size,
        "lr": args.lr, "question_max_length": 48, "passage_max_length": 192,
        "drop_last": True, "gradient_checkpointing": args.gradient_checkpointing,
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "device": str(device), "torch_version": torch.__version__,
        "training_seconds": elapsed, "cuda_peak_allocated_gb": peak_gb,
        "average_loss": loss_sum / steps, "checkpoint": str(output_pt),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_pt)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss", "query_exposures"])
        writer.writeheader()
        writer.writerows(logs)
    output_json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("Average loss:", metadata["average_loss"])
    print("Training seconds:", elapsed)
    print("CUDA peak allocated GB:", peak_gb)
    print("Saved:", output_pt)
    print("Metadata:", output_json)
    print("Step log:", output_csv)


if __name__ == "__main__":
    main()
