
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


MODEL_NAME = "distilbert-base-uncased"


class DPRDataset(Dataset):
    def __init__(self, split="train", max_samples=1000):

        dataset = load_dataset("squad", split=split)

        # Keep one question per unique context.
        # This prevents identical positive passages
        # from appearing in the same batch.
        self.examples = []
        seen_contexts = set()

        for example in dataset:
            context = example["context"]

            if context in seen_contexts:
                continue

            seen_contexts.add(context)

            
            self.examples.append({
                "question": example["question"],
                "passage": context,
                "answers": example["answers"]["text"]
            })


            if len(self.examples) >= max_samples:
                break

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def create_dataloader(
    split="train",
    max_samples=1000,
    batch_size=8,
    shuffle=True
):
    dataset = DPRDataset(
        split=split,
        max_samples=max_samples
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def collate_fn(batch):
        questions = [item["question"] for item in batch]
        passages = [item["passage"] for item in batch]

        question_inputs = tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=48,
            return_tensors="pt"
        )

        passage_inputs = tokenizer(
            passages,
            padding=True,
            truncation=True,
            max_length=192,
            return_tensors="pt"
        )

        return question_inputs, passage_inputs

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )


if __name__ == "__main__":
    loader = create_dataloader(
        max_samples=32,
        batch_size=4
    )

    question_inputs, passage_inputs = next(iter(loader))

    print("Question input shape:",
          question_inputs["input_ids"].shape)

    print("Passage input shape:",
          passage_inputs["input_ids"].shape)

    print("Attention mask shape:",
          question_inputs["attention_mask"].shape)

    print("\nFirst question tokens:")
    print(question_inputs["input_ids"][0])
