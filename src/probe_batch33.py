
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer

from src.model import DPR, MODEL_NAME
from src.data import DPRDataset
from src.loss import contrastive_loss


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    dataset = DPRDataset(
        split="train",
        max_samples=33
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    questions = [x["question"] for x in dataset.examples]
    passages = [x["passage"] for x in dataset.examples]

    def tokenize(texts, max_length):
        return tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

    q_inputs = tokenize(questions, 48)
    p_inputs = tokenize(passages, 192)

    model = DPR()

    model.question_encoder.gradient_checkpointing_enable()
    model.passage_encoder.gradient_checkpointing_enable()

    checkpoint = torch.load(
        "dpr_phase1.pt",
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(checkpoint)
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=2e-5)
    optimizer.zero_grad()

    q, p = model(q_inputs, p_inputs)

    loss, scores = contrastive_loss(q, p)

    print("Scores shape:", scores.shape)
    print("Loss:", loss.item())

    loss.backward()
    optimizer.step()

    print("One batch-33 optimizer step completed")


if __name__ == "__main__":
    main()
