
import torch
from transformers import AutoTokenizer

from src.model import DPR, MODEL_NAME
from src.data import DPRDataset

import torch.nn.functional as F


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("Device:", device)

    # 1. Load the training dataset
    dataset = DPRDataset(
        split="train",
        max_samples=1000
    )

    passages = [
        example["passage"] for example in dataset.examples
    ]

    questions = [
        example["question"] for example in dataset.examples[:128]
    ]

    print("Number of questions:", len(questions))


    print("Number of passages:", len(passages))

    # 2. Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    context = dataset.examples[0]["passage"]

    tokens = tokenizer(
        context,
        truncation=True,
        max_length=192
    )

    visible_text = tokenizer.decode(
        tokens["input_ids"],
        skip_special_tokens=True
    )

    print("\nORIGINAL CONTEXT LENGTH:")
    print(len(tokenizer.encode(context, truncation=False)))

    print("\nTRUNCATED CONTEXT:")
    print(visible_text)

    # 3. Initialize the dual encoder
    model = DPR().to(device)

    # 4. Restore the weights from our training run
    checkpoint = torch.load(
        "dpr_phase1.pt",
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint)

    # 5. Switch to evaluation mode (inference mode)
    model.eval()

    # 6. Encode the entire passage corpus
    passage_embeddings = []

    with torch.no_grad():
        for start in range(0, len(passages), 8):
            batch = passages[start:start + 8]

            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=192,
                return_tensors="pt"
            ).to(device)

            embeddings = model.encode_passage(
                inputs["input_ids"],
                inputs["attention_mask"]
            )

            # Move embeddings to CPU
            passage_embeddings.append(embeddings.cpu())

    P = torch.cat(passage_embeddings, dim=0)

    print("Passage embedding matrix shape:", P.shape)

    # 7. Store embeddings for our next step
    torch.save(
        {
            "passages": passages,
            "embeddings": P
        },
        "passage_index.pt"
    )

    print("Saved passage embeddings to passage_index.pt")


    # 8. Encode the training questions
    question_embeddings = []

    with torch.no_grad():
        for start in range(0, len(questions), 8):
            batch = questions[start:start + 8]

            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=48,
                return_tensors="pt"
            ).to(device)

            embeddings = model.encode_question(
                inputs["input_ids"],
                inputs["attention_mask"]
            )

            question_embeddings.append(embeddings.cpu())

    Q = torch.cat(question_embeddings, dim=0)

    print("Question embedding matrix:", Q.shape)

    # 9. Compare every question against every passage
    scores = Q @ P.T


    # Analyze positive passage ranks for all queries

    # Sort all passage IDs by descending similarity
    ranked_ids = torch.argsort(
        scores,
        dim=1,
        descending=True
    )

    # Positive passage i is located at index i
    positive_ids = torch.arange(len(questions))

    # Locate each positive passage in the ranked results
    matches = ranked_ids == positive_ids.unsqueeze(1)

    # Convert zero-based positions into one-based ranks
    positive_ranks = matches.int().argmax(dim=1) + 1

    # Extract positive similarity scores
    positive_scores = scores[
        torch.arange(len(questions)),
        positive_ids
    ]

    # Exclude known positives for margin calculation
    negative_scores = scores.clone()

    negative_scores[
        torch.arange(len(questions)),
        positive_ids
    ] = float("-inf")

    hardest_negative_scores = negative_scores.max(dim=1).values

    margins = positive_scores - hardest_negative_scores

    print("\n===== RETRIEVAL DIAGNOSTICS =====")

    print(
        "Average positive rank:",
        positive_ranks.float().mean().item()
    )

    print(
        "Median positive rank:",
        positive_ranks.float().median().item()
    )

    print(
        "Queries with positive ranked first:",
        (positive_ranks == 1).sum().item()
    )

    print(
        "Queries with positive outside top 5:",
        (positive_ranks > 5).sum().item()
    )

    print(
        "Queries with negative margins:",
        (margins < 0).sum().item()
    )

    print(
        "Average positive-negative margin:",
        margins.mean().item()
    )

    for k in [1, 5, 20, 100]:
        recall = (positive_ranks <= k).float().mean().item()

        print(f"Recall@{k}: {recall:.4f}")

    # Diagnose how well the model ranks the known positive

    query_id = 0
    positive_id = 0

    query_scores = scores[query_id].clone()

    positive_score = query_scores[positive_id].item()

    ranked_indices = torch.argsort(
        query_scores,
        descending=True
    )

    positive_rank = (
        (ranked_indices == positive_id)
        .nonzero(as_tuple=True)[0]
        .item() + 1
    )

    negative_scores = query_scores.clone()
    negative_scores[positive_id] = float("-inf")

    hardest_negative_score = negative_scores.max().item()

    margin = positive_score - hardest_negative_score

    print("\nPOSITIVE RANK DIAGNOSTIC")
    print("Positive passage score:", positive_score)
    print("Positive passage rank:", positive_rank)
    print("Hardest negative score:", hardest_negative_score)
    print("Positive-negative margin:", margin)


    # Diagnostic: compare dot product and cosine similarity

    q_norms = torch.linalg.vector_norm(Q, dim=1)
    p_norms = torch.linalg.vector_norm(P, dim=1)

    print("\nEmbedding norms:")
    print("Average question norm:", q_norms.mean().item())
    print("Average passage norm:", p_norms.mean().item())
    print("Min passage norm:", p_norms.min().item())
    print("Max passage norm:", p_norms.max().item())

    cosine_scores = (
        F.normalize(Q, p=2, dim=1)
        @ F.normalize(P, p=2, dim=1).T
    )

    # Compare both ranking strategies for query 0
    query_id = 0

    dot_scores = scores[query_id].clone()
    cos_scores = cosine_scores[query_id].clone()

    # Exclude the known positive passage
    dot_scores[query_id] = float("-inf")
    cos_scores[query_id] = float("-inf")

    dot_top = torch.topk(dot_scores, k=4)
    cos_top = torch.topk(cos_scores, k=4)

    print("\nTop 4 by dot product:")
    for idx, value in zip(dot_top.indices, dot_top.values):
        print(
            f"Passage {idx.item()} | "
            f"Score {value.item():.4f} | "
            f"Norm {p_norms[idx].item():.4f}"
        )

    print("\nTop 4 by cosine similarity:")
    for idx, value in zip(cos_top.indices, cos_top.values):
        print(
            f"Passage {idx.item()} | "
            f"Score {value.item():.4f} | "
            f"Norm {p_norms[idx].item():.4f}"
        )


    print("Similarity matrix:", scores.shape)

    # 10. Exclude each question's known positive passage
    positive_ids = torch.arange(len(questions))

    scores[
        torch.arange(len(questions)),
        positive_ids
    ] = float("-inf")

    # Retrieve more candidates than we need
    candidate_values, candidate_indices = torch.topk(
        scores,
        k=50,
        dim=1
    )

    # Collect answer-aware hard negatives
    filtered_indices = []
    filtered_scores = []

    for query_id in range(len(questions)):
        answers = dataset.examples[query_id]["answers"]

        selected_ids = []
        selected_scores = []

        for passage_id, score in zip(
            candidate_indices[query_id],
            candidate_values[query_id]
        ):
            passage_id = passage_id.item()
            passage_text = passages[passage_id].lower()

            # Reject passages containing an annotated answer
            contains_answer = any(
                answer.strip().lower() in passage_text
                for answer in answers
                if answer.strip()
            )

            if contains_answer:
                continue

            selected_ids.append(passage_id)
            selected_scores.append(score.item())

            if len(selected_ids) == 4:
                break

        if len(selected_ids) != 4:
            raise ValueError(
                f"Only found {len(selected_ids)} negatives "
                f"for query {query_id}. Increase candidate pool."
            )

        filtered_indices.append(selected_ids)
        filtered_scores.append(selected_scores)

    indices = torch.tensor(filtered_indices, dtype=torch.long)
    values = torch.tensor(filtered_scores, dtype=torch.float32)

    print("Filtered hard-negative indices:", indices.shape)

    print("Hard negative indices shape:", indices.shape)

    # 12. Inspect the first question
    query_id = 0

    print("\nQUESTION:")
    print(questions[query_id])

    print("\nPOSITIVE PASSAGE:")
    print(passages[query_id][:400])

    print("\nMINED HARD NEGATIVE CANDIDATES:")

    for rank, passage_id in enumerate(indices[query_id], start=1):
        passage_id = passage_id.item()

        print(f"\nRank {rank}")
        print("Passage ID:", passage_id)
        print("Similarity:", values[query_id, rank - 1].item())
        print("Passage:", passages[passage_id][:400])

    # 13. Save mining results
    torch.save(
        {
            "questions": questions,
            "passages": passages,
            "hard_negative_indices": indices,
            "hard_negative_scores": values
        },
        "mined_negatives.pt"
    )

    print("\nSaved mining results to mined_negatives.pt")


if __name__ == "__main__":
    main()
