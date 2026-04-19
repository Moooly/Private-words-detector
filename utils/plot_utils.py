import json
import matplotlib.pyplot as plt


def compute_best_epoch_with_min_delta(values, min_delta=0.005):
    if len(values) == 0:
        return None

    best_value = float(values[0])
    best_epoch_idx = 0

    for i in range(1, len(values)):
        current = float(values[i])
        if current < best_value - min_delta:
            best_value = current
            best_epoch_idx = i

    return best_epoch_idx + 1


def save_training_outputs(history):
    history_dict = history.history

    train_loss = history_dict.get("loss", [])
    val_loss = history_dict.get("val_loss", [])
    train_acc = history_dict.get("sparse_categorical_accuracy", [])
    val_acc = history_dict.get("val_sparse_categorical_accuracy", [])

    epochs_range = list(range(1, len(train_loss) + 1))
    best_epoch = compute_best_epoch_with_min_delta(val_loss, min_delta=0.005)

    history_to_save = {
        "epoch": epochs_range,
        "train_loss": [float(x) for x in train_loss],
        "val_loss": [float(x) for x in val_loss],
        "train_accuracy": [float(x) for x in train_acc],
        "val_accuracy": [float(x) for x in val_acc],
        "best_epoch_val_loss": best_epoch,
    }

    with open("training_history.json", "w", encoding="utf-8") as f:
        json.dump(history_to_save, f, indent=2)
    print("Saved: training_history.json")

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, train_loss, marker="o", label="Training Loss")
    if len(val_loss) > 0:
        plt.plot(epochs_range, val_loss, marker="o", label="Validation Loss")

    if best_epoch is not None:
        plt.axvline(
            x=best_epoch,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Best Epoch = {best_epoch}"
        )

    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(epochs_range)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("training_validation_loss.png", dpi=350, bbox_inches="tight")
    plt.close()
    print("Saved: training_validation_loss.png")

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, train_acc, marker="o", label="Training Accuracy")
    if len(val_acc) > 0:
        plt.plot(epochs_range, val_acc, marker="o", label="Validation Accuracy")

    if best_epoch is not None:
        plt.axvline(
            x=best_epoch,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Best Epoch = {best_epoch}"
        )

    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(epochs_range)
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("training_validation_accuracy.png", dpi=350, bbox_inches="tight")
    plt.close()
    print("Saved: training_validation_accuracy.png")