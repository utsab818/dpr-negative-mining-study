import argparse
import random
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer

from src.data import DPRDataset
from src.loss import contrastive_loss, hard_negative_loss, mixed_negative_loss
from src.model import DPR, MODEL_NAME


MINED_PATH = Path("mined_negatives.pt")
WARMUP_PATH = Path("dpr_phase1.pt")
OUTPUT_DIR = Path("experiments/v2/checkpoints")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def tokenize(tokenizer, texts, max_length, device):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)


def batch_id_generator(num_examples, batch_size, seed):
    """Produce shuffled complete batches, reshuffling between epochs."""
    if batch_size < 1 or batch_size > num_examples:
        raise ValueError("Batch size must be between 1 and dataset size")

    generator = torch.Generator().manual_seed(seed)
    while True:
        order = torch.randperm(num_examples, generator=generator).tolist()
        for start in range(0, num_examples - batch_size + 1, batch_size):
            yield order[start : start + batch_size]


def train(args):
    set_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)
    print("Strategy:", args.strategy)
    print("Seed:", args.seed)

    mined = torch.load(MINED_PATH, map_location="cpu", weights_only=True)
    questions = mined["questions"]
    passages = mined["passages"]
    hard_ids = mined["hard_negative_indices"]

    num_examples = len(questions)
    if num_examples == 0 or num_examples != hard_ids.size(0):
        raise ValueError("Questions and mined negatives do not align")
    if args.batch_size > num_examples:
        raise ValueError("Batch size cannot exceed number of training queries")
    if args.strategy in ("hard", "mixed") and args.num_negatives > hard_ids.size(1):
        raise ValueError("Not enough mined hard negatives for requested K")

    dataset = DPRDataset(split="train", max_samples=num_examples)
    if len(dataset.examples) != num_examples or any(
        dataset.examples[i]["question"] != questions[i]
        or dataset.examples[i]["passage"] != passages[i]
        for i in range(num_examples)
    ):
        raise ValueError("Training examples do not match the mining artifact")

    # Construct random candidates once, rather than reloading the dataset per step.
    random_candidates = None
    if args.strategy == "random":
        random_candidates = []
        folded_passages = [passage.casefold() for passage in passages]
        for i, example in enumerate(dataset.examples):
            answers = [
                a.strip().casefold() for a in example["answers"] if a.strip()
            ]
            eligible = [
                j for j, passage in enumerate(folded_passages)
                if j != i and not any(answer in passage for answer in answers)
            ]
            if len(eligible) < args.num_negatives:
                raise ValueError(f"Not enough random negatives for query {i}")
            random_candidates.append(eligible)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = DPR()
    checkpoint = torch.load(WARMUP_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint)

    if args.strategy == "inbatch":
        model.question_encoder.gradient_checkpointing_enable()
        model.passage_encoder.gradient_checkpointing_enable()

    model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=args.lr)

    batch_generator = batch_id_generator(num_examples, args.batch_size, args.seed)
    negative_rng = random.Random(args.seed)

    target_steps = args.max_steps
    if target_steps is None:
        target_steps = num_examples // args.batch_size

    output = OUTPUT_DIR / (
        f"dpr_{args.strategy}_k{args.num_negatives}_"
        f"seed{args.seed}_steps{target_steps}.pt"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Checkpoint already exists: {output}")

    total_loss = 0.0
    for step in range(1, target_steps + 1):
        batch_ids = next(batch_generator)
        batch_questions = [questions[i] for i in batch_ids]
        batch_positives = [passages[i] for i in batch_ids]

        q_inputs = tokenize(tokenizer, batch_questions, 48, device)
        pos_inputs = tokenize(tokenizer, batch_positives, 192, device)

        neg_inputs = None
        if args.strategy != "inbatch":
            negative_ids = []
            for i in batch_ids:
                if args.strategy == "random":
                    selected = negative_rng.sample(
                        random_candidates[i], args.num_negatives
                    )
                else:
                    selected = hard_ids[i, : args.num_negatives].tolist()
                negative_ids.extend(selected)
            batch_negatives = [passages[j] for j in negative_ids]
            neg_inputs = tokenize(tokenizer, batch_negatives, 192, device)

        optimizer.zero_grad()
        q = model.encode_question(q_inputs["input_ids"], q_inputs["attention_mask"])
        p_pos = model.encode_passage(
            pos_inputs["input_ids"], pos_inputs["attention_mask"]
        )

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
        total_loss += loss.item()

        if step == 1 or step % 5 == 0 or step == target_steps:
            print(
                f"Step {step} | Scores {tuple(scores.shape)} | "
                f"Loss {loss.item():.4f}"
            )

    print("Total optimizer steps:", target_steps)
    print("Query exposures:", target_steps * args.batch_size)
    print("Average loss:", total_loss / target_steps)
    torch.save(model.state_dict(), output)
    print("Saved:", output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy", choices=["random", "hard", "mixed", "inbatch"], required=True
    )
    parser.add_argument("--num-negatives", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    if args.num_negatives < 1 or args.batch_size < 1:
        parser.error("--num-negatives and --batch-size must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.strategy == "inbatch" and args.batch_size != args.num_negatives + 1:
        parser.error("In-batch requires --batch-size = --num-negatives + 1")
    if args.strategy == "mixed" and args.batch_size < 2:
        parser.error("Mixed requires --batch-size >= 2")

    train(args)


if __name__ == "__main__":
    main()
