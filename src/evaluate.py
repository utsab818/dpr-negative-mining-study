
import torch
from transformers import AutoTokenizer

from src.model import DPR, MODEL_NAME
from src.data import DPRDataset

import csv
import os


def evaluate(checkpoint_path):
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Load held-out SQuAD validation examples
    dataset = DPRDataset(
        split="validation",
        max_samples=1000
    )

    questions = [
        example["question"] for example in dataset.examples
    ]
    passages = [
        example["passage"] for example in dataset.examples
    ]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = DPR().to(device)

    # Load the weights saved during training
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
)
    model.load_state_dict(state_dict)
    model.eval()

    def encode_texts(texts, encode_fn, max_length):
        embeddings = []

        with torch.no_grad():
            for start in range(0, len(texts), 8):
                batch = texts[start:start + 8]

                inputs = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                ).to(device)

                vectors = encode_fn(
                    inputs["input_ids"],
                    inputs["attention_mask"]
                )

                embeddings.append(vectors.cpu())

        return torch.cat(embeddings, dim=0)

    # Encode all queries and candidate passages
    Q = encode_texts(
        questions,
        model.encode_question,
        max_length=48
    )

    P = encode_texts(
        passages,
        model.encode_passage,
        max_length=192
    )

    # Similarity scores for every query-passage pair
    scores = Q @ P.T

    print("Similarity matrix shape:", scores.shape)

    # Rank all passages for each question
    rankings = torch.argsort(
        scores,
        dim=1,
        descending=True
    )

    
    # Positive passage i corresponds to question i
    correct_ids = torch.arange(len(questions))

    metrics = {}

    for k in [1, 5, 20]:
        top_k = rankings[:, :k]

        hits = (
            top_k == correct_ids.unsqueeze(1)
        ).any(dim=1)

        recall = hits.float().mean().item()

        metrics[f"recall@{k}"] = recall
        print(f"Recall@{k}: {recall:.4f}")

    # MRR@10
    top_10 = rankings[:, :10]
    matches = top_10 == correct_ids.unsqueeze(1)

    reciprocal_ranks = torch.zeros(len(questions))

    for i in range(len(questions)):
        found = torch.where(matches[i])[0]

        if len(found) > 0:
            reciprocal_ranks[i] = 1.0 / (found[0].item() + 1)

    metrics["mrr@10"] = reciprocal_ranks.mean().item()

    print(f"MRR@10: {metrics['mrr@10']:.4f}")

    return metrics




if __name__ == "__main__":
    
    checkpoints = {
        "warmup": "experiments/v1/checkpoints/dpr_phase1.pt",

        "random_k4_steps32": (
            "experiments/v2/checkpoints/"
            "dpr_random_k4_seed42_steps32.pt"
        ),

        "hard_k4_steps32": (
            "experiments/v2/checkpoints/"
            "dpr_hard_k4_seed42_steps32.pt"
        ),

        "mixed_k4_steps32": (
            "experiments/v2/checkpoints/"
            "dpr_mixed_k4_seed42_steps32.pt"
        ),

        "inbatch_k32_steps32": (
            "experiments/v2/checkpoints/"
            "dpr_inbatch_k32_seed42_steps32.pt"
        ),

        "inbatch_k32_steps4": (
            "experiments/v2/checkpoints/"
            "dpr_inbatch_k32_seed42_steps4.pt"
        )
    }


    results = []

    for strategy, checkpoint_path in checkpoints.items():
        print(f"\n===== {strategy.upper()} =====")

        metrics = evaluate(checkpoint_path)

        results.append({
            "strategy": strategy,
            "seed": 42,
            "checkpoint": checkpoint_path,
            **metrics
        })

    os.makedirs("results", exist_ok=True)
    os.makedirs("experiments/v2/results", exist_ok=True)

    # output_path = "results/phase3_seed42.csv"
    # output_path = "results/phase3_seed42_1000.csv"
    # output_path = "results/phase3_seed42_1000_with_inbatch.csv"
    output_path = "experiments/v2/results/seed42_1000.csv"

    if os.path.exists(output_path):
        raise FileExistsError(
            f"Results already exist: {output_path}"
        )

    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "strategy",
                "seed",
                "checkpoint",
                "recall@1",
                "recall@5",
                "recall@20",
                "mrr@10"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_path}")

