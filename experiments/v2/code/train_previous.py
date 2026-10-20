
import torch
from torch.optim import AdamW

from src.model import DPR
from src.loss import contrastive_loss
from src.data import DPRDataset, create_dataloader

from transformers import AutoTokenizer
from src.model import MODEL_NAME
from src.loss import hard_negative_loss

import random
from src.loss import mixed_negative_loss


def train():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("Training device:", device)

    # Load question-positive passage pairs
    dataloader = create_dataloader(
        split="train",
        max_samples=128,
        batch_size=4,
        shuffle=True
    )

    # Initialize our two independent encoders
    model = DPR().to(device)
    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=2e-5
    )

    epochs = 1

    for epoch in range(epochs):
        total_loss = 0.0

        for step, (question_inputs, passage_inputs) in enumerate(dataloader):

            question_inputs = question_inputs.to(device)
            passage_inputs = passage_inputs.to(device)

            # Clear previous gradients
            optimizer.zero_grad()

            # Encode questions and passages
            q, p = model(question_inputs, passage_inputs)

            # Calculate in-batch contrastive loss
            loss, scores = contrastive_loss(q, p)

            # Calculate gradients
            loss.backward()

            # Update encoder parameters
            optimizer.step()

            total_loss += loss.item()

            if step % 5 == 0:
                print(
                    f"Epoch {epoch + 1} | "
                    f"Step {step} | "
                    f"Loss {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)

        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

    # Save trained model parameters
    torch.save(
        model.state_dict(),
        "dpr_phase1.pt"
    )

    print("Model saved to dpr_phase1.pt")


def train_hard_negatives():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("Training device:", device)

    # Load mined passages and their indices
    mined = torch.load(
        "mined_negatives.pt",
        map_location="cpu",
        weights_only=True
    )

    questions = mined["questions"]
    passages = mined["passages"]
    negative_ids = mined["hard_negative_indices"]

    batch_size = 4
    num_negatives = negative_ids.size(1)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Start from the SAME warm-up checkpoint
    model = DPR().to(device)

    state_dict = torch.load(
        "dpr_phase1.pt",
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=2e-5
    )

    total_loss = 0.0
    steps = 0

    # Shuffle query indices, not individual passages
    order = torch.randperm(len(questions))

    for start in range(0, len(questions), batch_size):
        batch_ids = order[start:start + batch_size]

        # Keep full batches for consistent tensor dimensions
        if len(batch_ids) < batch_size:
            continue

        batch_questions = [
            questions[i] for i in batch_ids.tolist()
        ]

        # Positive passage i corresponds to question i
        batch_positives = [
            passages[i] for i in batch_ids.tolist()
        ]

        # Four mined negatives for each question
        batch_negatives = [
            passages[negative_ids[i, j].item()]
            for i in batch_ids.tolist()
            for j in range(num_negatives)
        ]

        question_inputs = tokenizer(
            batch_questions,
            padding=True,
            truncation=True,
            max_length=48,
            return_tensors="pt"
        ).to(device)

        positive_inputs = tokenizer(
            batch_positives,
            padding=True,
            truncation=True,
            max_length=192,
            return_tensors="pt"
        ).to(device)

        negative_inputs = tokenizer(
            batch_negatives,
            padding=True,
            truncation=True,
            max_length=192,
            return_tensors="pt"
        ).to(device)

        optimizer.zero_grad()

        q = model.encode_question(
            question_inputs["input_ids"],
            question_inputs["attention_mask"]
        )

        p_pos = model.encode_passage(
            positive_inputs["input_ids"],
            positive_inputs["attention_mask"]
        )

        p_neg = model.encode_passage(
            negative_inputs["input_ids"],
            negative_inputs["attention_mask"]
        )

        # [B*K, D] -> [B, K, D]
        p_neg = p_neg.reshape(
            batch_size,
            num_negatives,
            -1
        )

        loss, scores = hard_negative_loss(
            q,
            p_pos,
            p_neg
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1

        if steps % 5 == 0:
            print(
                f"Step {steps} | "
                f"Loss {loss.item():.4f}"
            )

    print(
        "Average hard-negative loss:",
        total_loss / steps
    )

    torch.save(
        model.state_dict(),
        "dpr_hard_only.pt"
    )

    print("Saved model to dpr_hard_only.pt")


def train_random_negatives():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    random.seed(42)
    torch.manual_seed(42)

    mined = torch.load(
        "mined_negatives.pt",
        map_location="cpu",
        weights_only=True
    )

    questions = mined["questions"]
    passages = mined["passages"]

    dataset = DPRDataset(split="train", max_samples=128)
    answers_by_query = [
        example["answers"] for example in dataset.examples
    ]

    batch_size = 4
    num_negatives = 4

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = DPR().to(device)

    checkpoint = torch.load(
        "dpr_phase1.pt",
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint)
    model.train()

    optimizer = AdamW(model.parameters(), lr=2e-5)

    order = torch.randperm(len(questions))

    total_loss = 0.0
    steps = 0

    for start in range(0, len(questions), batch_size):
        batch_ids = order[start:start + batch_size].tolist()

        if len(batch_ids) < batch_size:
            continue

        batch_questions = [questions[i] for i in batch_ids]
        batch_positives = [passages[i] for i in batch_ids]

        batch_negatives = []

        for query_id in batch_ids:
            # Exclude positive and known answer-containing passages
            answers = answers_by_query[query_id]

            eligible = [
                j for j, passage in enumerate(passages)
                if j != query_id
                and not any(
                    answer.strip().lower() in passage.lower()
                    for answer in answers
                    if answer.strip()
                )
            ]

            sampled_ids = random.sample(eligible, num_negatives)

            batch_negatives.extend(
                passages[j] for j in sampled_ids
            )

        def tokenize(texts, max_length):
            return tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)

        q_inputs = tokenize(batch_questions, 48)
        pos_inputs = tokenize(batch_positives, 192)
        neg_inputs = tokenize(batch_negatives, 192)

        optimizer.zero_grad()

        q = model.encode_question(
            q_inputs["input_ids"],
            q_inputs["attention_mask"]
        )

        p_pos = model.encode_passage(
            pos_inputs["input_ids"],
            pos_inputs["attention_mask"]
        )

        p_neg = model.encode_passage(
            neg_inputs["input_ids"],
            neg_inputs["attention_mask"]
        ).reshape(batch_size, num_negatives, -1)

        loss, _ = hard_negative_loss(q, p_pos, p_neg)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1

        if steps % 5 == 0:
            print(f"Step {steps} | Loss {loss.item():.4f}")

    print("Average random-negative loss:", total_loss / steps)

    torch.save(model.state_dict(), "dpr_random_only.pt")

    print("Saved model to dpr_random_only.pt")


def train_mixed_negatives():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("Training device:", device)

    # Reproducible initial random state
    torch.manual_seed(42)

    mined = torch.load(
        "mined_negatives.pt",
        map_location="cpu",
        weights_only=True
    )

    questions = mined["questions"]
    passages = mined["passages"]
    negative_ids = mined["hard_negative_indices"]

    batch_size = 4
    num_negatives = negative_ids.size(1)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = DPR().to(device)

    # Start from the original warm-up checkpoint
    checkpoint = torch.load(
        "dpr_phase1.pt",
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint)
    model.train()

    optimizer = AdamW(model.parameters(), lr=2e-5)

    total_loss = 0.0
    steps = 0

    order = torch.randperm(len(questions))

    for start in range(0, len(questions), batch_size):
        batch_ids = order[start:start + batch_size].tolist()

        if len(batch_ids) < batch_size:
            continue

        batch_questions = [questions[i] for i in batch_ids]
        batch_positives = [passages[i] for i in batch_ids]

        batch_negatives = [
            passages[negative_ids[i, j].item()]
            for i in batch_ids
            for j in range(num_negatives)
        ]

        def tokenize(texts, max_length):
            return tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)

        q_inputs = tokenize(batch_questions, 48)
        pos_inputs = tokenize(batch_positives, 192)
        neg_inputs = tokenize(batch_negatives, 192)

        optimizer.zero_grad()

        q = model.encode_question(
            q_inputs["input_ids"],
            q_inputs["attention_mask"]
        )

        p_pos = model.encode_passage(
            pos_inputs["input_ids"],
            pos_inputs["attention_mask"]
        )

        p_neg = model.encode_passage(
            neg_inputs["input_ids"],
            neg_inputs["attention_mask"]
        ).reshape(batch_size, num_negatives, -1)

        loss, scores = mixed_negative_loss(
            q,
            p_pos,
            p_neg
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1

        if steps % 5 == 0:
            print(
                f"Step {steps} | "
                f"Loss {loss.item():.4f}"
            )

    print(
        "Average mixed-negative loss:",
        total_loss / steps
    )

    torch.save(
        model.state_dict(),
        "dpr_mixed.pt"
    )

    print("Saved model to dpr_mixed.pt")


# if __name__ == "__main__":
#     train()

# if __name__ == "__main__":
#     train_hard_negatives()

# if __name__ == "__main__":
#     train_random_negatives()

if __name__ == "__main__":
    train_mixed_negatives()