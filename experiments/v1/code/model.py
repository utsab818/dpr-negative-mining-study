
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "distilbert-base-uncased"


class DPR(nn.Module):
    def __init__(self):
        super().__init__()

        # Two independent encoders
        self.question_encoder = AutoModel.from_pretrained(MODEL_NAME)
        self.passage_encoder = AutoModel.from_pretrained(MODEL_NAME)

    def encode_question(self, input_ids, attention_mask):
        outputs = self.question_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # First token representation, for [CLS] pooling. We could also use mean pooling.
        return outputs.last_hidden_state[:, 0, :]

    def encode_passage(self, input_ids, attention_mask):
        outputs = self.passage_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        return outputs.last_hidden_state[:, 0, :]

    def forward(self, question_inputs, passage_inputs):
        q = self.encode_question(
            question_inputs["input_ids"],
            question_inputs["attention_mask"]
        )

        p = self.encode_passage(
            passage_inputs["input_ids"],
            passage_inputs["attention_mask"]
        )

        return q, p


if __name__ == "__main__":
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = DPR().to(device)

    question = ["Who invented the telephone?"]
    passage = [
        "Alexander Graham Bell is credited with inventing the telephone."
    ]

    question_inputs = tokenizer(
        question,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    passage_inputs = tokenizer(
        passage,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    model.eval()

    with torch.no_grad():
        q, p = model(question_inputs, passage_inputs)

        print("Question embedding shape:", q.shape)
        print("Passage embedding shape:", p.shape)
        print("Similarity score:", torch.sum(q * p).item())
