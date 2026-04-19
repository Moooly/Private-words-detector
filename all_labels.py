import csv
import os
from datasets import load_dataset


def main():
    train_size = 0.7
    test_size = 0.3
    seed = 42

    ds_full = load_dataset("Isotonic/pii-masking-200k", split="train").shuffle(seed=seed)
    total_size = len(ds_full)

    train_end = int(total_size * train_size)
    test_end = int(total_size * (train_size + test_size))

    train_dataset = ds_full.select(range(0, train_end))
    test_dataset = ds_full.select(range(train_end, test_end))

    train_bio_labels = []
    for row in train_dataset:
        labels = row["bio_labels"]
        if labels is not None:
            train_bio_labels.append(labels)

    test_bio_labels = []
    for row in test_dataset:
        labels = row["bio_labels"]
        if labels is not None:
            test_bio_labels.append(labels)

    all_label_sequences = train_bio_labels + test_bio_labels
    unique_labels = sorted({label for seq in all_label_sequences for label in seq})

    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/all_labels_in_dataset.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label_id", "label"])
        for idx, label in enumerate(unique_labels):
            writer.writerow([idx, label])

    print("All labels in dataset:")
    for idx, label in enumerate(unique_labels):
        print(f"{idx}\t{label}")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()