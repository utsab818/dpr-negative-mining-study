
import argparse
import random

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer

from src.model import DPR, MODEL_NAME
from src.data import DPRDataset
from src.loss import (
    contrastive_loss,
    hard_negative_loss,
    mixed_negative_loss
)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def tokenize(tokenizer, texts, max_length, device):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)


def batch_id_generator(
    num_examples,
    batch_size,
    seed
):
    generator = torch.Generator().manual_seed(seed)

    if batch_size > num_examples:
        raise ValueError(
            "Batch size cannot exceed dataset size"
        )

    while True:
        order = torch.randperm(
            num_examples,
            generator=generator
        ).tolist()

        # Use only complete batches
        for start in range(
            0,
            num_examples - batch_size + 1,
            batch_size
        ):
            yield order[start:start + batch_size]


def train(args):
    set_seed(args.seed)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("Device:", device)
    print("Strategy:", args.strategy)
    print("Seed:", args.seed)

    # Load the mined training collection
    mined = torch.load(
        "mined_negatives.pt",
        map_location="cpu",
        weights_only=True
    )

    questions = mined["questions"]
    passages = mined["passages"]
    hard_ids = mined["hard_negative_indices"]

    assert len(questions) == hard_ids.size(0)

    # Recover answers for false-negative filtering
    dataset = DPRDataset(
        split="train",
        max_samples=len(questions)
    )

    assert all(
        dataset.examples[i]["question"] == questions[i]
        and dataset.examples[i]["passage"] == passages[i]
        for i in range(len(questions))
    ), "Training examples do not match mined data"

    answers_by_query = [
        example["answers"] for example in dataset.examples
    ]

    # Precompute eligible random negatives
    random_candidates = []

    for i, answers in enumerate(answers_by_query):
        eligible = [
            j for j, passage in enumerate(passages)
            if j != i
            and not any(
                answer.strip().casefold() in passage.casefold()
                for answer in answers
                if answer.strip()
            )
        ]

        if len(eligible) < args.num_negatives:
            raise ValueError("Not enough random negatives")

        random_candidates.append(eligible)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Every strategy starts from identical weights
    model = DPR().to(device)

    checkpoint = torch.load(
        "dpr_phase1.pt",
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint)

    if args.strategy == "inbatch":
        model.question_encoder.gradient_checkpointing_enable()
        model.passage_encoder.gradient_checkpointing_enable()

    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr
    )

    # Separate generators for reproducibility
    generator = torch.Generator().manual_seed(args.seed)
    rng = random.Random(args.seed)

    order = torch.randperm(
        len(questions),
        generator=generator
    ).tolist()

    
    total_loss = 0.0
    steps = 0

    target_steps = args.max_steps

    if target_steps is None:
        target_steps = len(questions) // args.batch_size

    while steps < target_steps:

        # Generate a shuffled full batch.
        # When the dataset is exhausted, reshuffle.
        # Sampling is performed without replacement
        # within each batch.
        batch_generator = batch_id_generator(
            num_examples=len(questions),
            batch_size=args.batch_size,
            seed=args.seed
        )
        batch_ids = next(batch_generator)
        
        batch_questions = [
            questions[i] for i in batch_ids
        ]

        batch_positives = [
            passages[i] for i in batch_ids
        ]

        # Select negatives according to strategy
        batch_negatives = []

        if args.strategy != "inbatch":
            for query_id in batch_ids:

                if args.strategy == "random":
                    selected = rng.sample(
                        random_candidates[query_id],
                        args.num_negatives
                    )
                else:
                    selected = hard_ids[
                        query_id, :args.num_negatives
                    ].tolist()

                batch_negatives.extend(
                    passages[j] for j in selected
                )

        
        q_inputs = tokenize(
            tokenizer, batch_questions, 48, device
        )

        pos_inputs = tokenize(
            tokenizer, batch_positives, 192, device
        )

        # In-batch training does not need explicit negatives
        if args.strategy != "inbatch":
            neg_inputs = tokenize(
                tokenizer, batch_negatives, 192, device
        )

        
        optimizer.zero_grad()

        q = model.encode_question(
            q_inputs["input_ids"],
            q_inputs["attention_mask"]
        )

        p_pos = model.encode_passage(
            pos_inputs["input_ids"],
            pos_inputs["attention_mask"]
        )

        if args.strategy == "inbatch":
            # Every other positive is an in-batch negative
            loss, scores = contrastive_loss(q, p_pos)

        else:
            neg_inputs = tokenize(
                tokenizer, batch_negatives, 192, device
            )

            p_neg = model.encode_passage(
                neg_inputs["input_ids"],
                neg_inputs["attention_mask"]
            )

            p_neg = p_neg.reshape(
                len(batch_ids),
                args.num_negatives,
                -1
            )

            if args.strategy == "mixed":
                loss, scores = mixed_negative_loss(
                    q, p_pos, p_neg
                )
            else:
                loss, scores = hard_negative_loss(
                    q, p_pos, p_neg
                )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1

        if steps == 1 or steps % 5 == 0:
            print(
                f"Step {steps} | "
                f"Scores {tuple(scores.shape)} | "
                f"Loss {loss.item():.4f}"
            )

    average_loss = total_loss / steps

    print("Total optimizer steps:", steps)
    print("Average loss:", average_loss)

    output = (
        f"dpr_{args.strategy}_"
        f"k{args.num_negatives}_seed{args.seed}.pt"
    )

    torch.save(model.state_dict(), output)
    print("Saved:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--strategy",
        choices=["random", "hard", "mixed", "inbatch"],
        required=True
    )

    parser.add_argument("--num-negatives", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-steps", type=int, default=None)

    args = parser.parse_args()

    if args.num_negatives < 1:
        parser.error("--num-negatives must be at least 1")

    if args.strategy in ["hard", "mixed"] and args.num_negatives > 4:
        parser.error("Only four mined negatives are currently saved")

    if args.strategy == "inbatch":
        if args.batch_size != args.num_negatives + 1:
            parser.error(
                "For inbatch training, batch-size must equal "
                "num-negatives + 1"
            )

    if args.strategy == "mixed" and args.batch_size < 2:
        parser.error("Mixed training needs batch-size >= 2")

    if args.strategy in ["hard", "mixed"] and args.num_negatives > 4:
        parser.error("Only four mined negatives are currently saved")

    if args.batch_size > 128:
        parser.error(
        "Batch size cannot exceed the 128 training queries"
        )

    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")

    train(args)
