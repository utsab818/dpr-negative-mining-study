
import torch
import torch.nn.functional as F


def contrastive_loss(question_embeddings, passage_embeddings):
    # [batch_size, 768] @ [768, batch_size]
    # Result: [batch_size, batch_size]
    scores = question_embeddings @ passage_embeddings.T

    # Correct passage for question i is passage i
    labels = torch.arange(
        scores.size(0),
        device=scores.device
    )

    # Softmax cross-entropy over passage scores
    loss = F.cross_entropy(scores, labels)

    return loss, scores


def hard_negative_loss(
    question_embeddings,
    positive_embeddings,
    negative_embeddings
):
    # Q: [B, D]
    # P+: [B, D]
    # P-: [B, K, D]

    # Dot product of each question with its positive
    positive_scores = (
        question_embeddings * positive_embeddings
    ).sum(dim=1, keepdim=True)

    # Dot product of each question with its K negatives
    negative_scores = torch.bmm(
        negative_embeddings,
        question_embeddings.unsqueeze(2)
    ).squeeze(2)

    # [B, 1] + [B, K] -> [B, K+1]
    scores = torch.cat(
        [positive_scores, negative_scores],
        dim=1
    )

    # Positive is always in column 0
    labels = torch.zeros(
        scores.size(0),
        dtype=torch.long,
        device=scores.device
    )

    loss = F.cross_entropy(scores, labels)

    return loss, scores

def mixed_negative_loss(
    question_embeddings,
    positive_embeddings,
    negative_embeddings
):
    # question_embeddings: [B, D]
    # positive_embeddings: [B, D]
    # negative_embeddings: [B, K, D]

    # In-batch scores: [B, B]
    batch_scores = (
        question_embeddings @ positive_embeddings.T
    )

    # Mined-negative scores: [B, K]
    hard_scores = torch.bmm(
        negative_embeddings,
        question_embeddings.unsqueeze(2)
    ).squeeze(2)

    # Combine both sets of scores: [B, B + K]
    scores = torch.cat(
        [batch_scores, hard_scores],
        dim=1
    )

    # Positive passage i is at column i
    labels = torch.arange(
        question_embeddings.size(0),
        device=scores.device
    )

    loss = F.cross_entropy(scores, labels)

    return loss, scores

if __name__ == "__main__":

    # 3 questions, each represented by 3 numbers
    questions = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    # 3 positive passages
    passages = torch.tensor([
        [3.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 3.0]
    ])

    # passages = torch.tensor([
    #         [0.0, 3.0, 0.0],
    #         [0.0, 3.0, 0.0],
    #         [0.0, 0.0, 3.0]
    #     ])

    loss, scores = contrastive_loss(questions, passages)

    print("Similarity matrix:")
    print(scores)

    print("\nTarget labels:")
    print(torch.arange(questions.size(0)))

    print("\nContrastive loss:")
    print(loss.item())

    print("\n=== Hard-negative loss test ===")

    questions = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0]
    ])

    positives = torch.tensor([
        [3.0, 0.0],
        [0.0, 3.0]
    ])

    negatives = torch.tensor([
        [
            [2.0, 0.0],
            [0.0, 1.0]
        ],
        [
            [0.0, 2.0],
            [1.0, 0.0]
        ]
    ])

    loss, scores = hard_negative_loss(
        questions,
        positives,
        negatives
    )

    print("Scores:")
    print(scores)

    print("Scores shape:", scores.shape)
    print("Hard-negative loss:", loss.item())


    print("\n=== Mixed-negative loss test ===")

    questions = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0]
    ])

    positives = torch.tensor([
        [3.0, 0.0],
        [0.0, 3.0]
    ])

    hard_negatives = torch.tensor([
        [
            [2.0, 0.0],
            [0.0, 1.0]
        ],
        [
            [0.0, 2.0],
            [1.0, 0.0]
        ]
    ])

    loss, scores = mixed_negative_loss(
        questions,
        positives,
        hard_negatives
    )

    print("Scores:")
    print(scores)

    print("Scores shape:", scores.shape)
    print("Mixed-negative loss:", loss.item())

