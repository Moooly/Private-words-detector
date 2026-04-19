import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from utils.visualization import (
    plot_confusion_matrices,
    plot_f1_top_bottom,
    plot_precision_recall_f1_grouped,
    plot_performance_by_sentence_length,
    plot_false_negative_false_positive_by_label,
    plot_bio_error_type_chart,
    print_top_confusions,
    plot_accuracy_summary,
    plot_sentence_length_distribution,
)


def evaluate_test_predictions(padded_X_word_test, padded_Y_test, pred_ids, label2id, num_classes):
    y_true = padded_Y_test.reshape(-1)
    y_pred = pred_ids.reshape(-1)
    nonpad = (padded_X_word_test.reshape(-1) != 0)

    o_id = int(label2id["O"])
    mask_no_pad = nonpad
    mask_no_pad_no_o = nonpad & (y_true != o_id)

    y_true_np = y_true[mask_no_pad]
    y_pred_np = y_pred[mask_no_pad]

    print("\n========== Classification Report (No PAD) ==========")
    labels_sorted = list(range(num_classes))

    if mask_no_pad_no_o.any():
        pii_true = y_true[mask_no_pad_no_o]
        pii_pred = y_pred[mask_no_pad_no_o]
        pii_macro_f1 = f1_score(pii_true, pii_pred, average="macro", zero_division=0)
        print("Macro-F1 (exclude PAD, exclude O):", float(pii_macro_f1))
    else:
        pii_true = np.array([], dtype=y_true.dtype)
        pii_pred = np.array([], dtype=y_pred.dtype)
        print("Macro-F1 (exclude PAD, exclude O): NA (No PII tokens)")

    if mask_no_pad_no_o.any():
        pii_acc = float((pii_pred == pii_true).mean())
    else:
        pii_acc = float("nan")

    print("Accuracy (exclude PAD, exclude O):", pii_acc)
    print("PII token count:", int(mask_no_pad_no_o.sum()))

    cm = confusion_matrix(y_true_np, y_pred_np, labels=labels_sorted)
    support_counts = np.bincount(y_true_np, minlength=num_classes)

    return {
        "y_true_np": y_true_np,
        "y_pred_np": y_pred_np,
        "o_id": o_id,
        "labels_sorted": labels_sorted,
        "cm": cm,
        "support_counts": support_counts,
    }


def run_visualizations(raw_all_tokens, raw_test_tokens_dropped, raw_test_bio_labels_dropped, pred_ids, label2id, id2label, num_classes, eval_bundle):
    o_id = eval_bundle["o_id"]
    y_true_np = eval_bundle["y_true_np"]
    y_pred_np = eval_bundle["y_pred_np"]
    labels_sorted = eval_bundle["labels_sorted"]
    cm = eval_bundle["cm"]
    support_counts = eval_bundle["support_counts"]

    plot_confusion_matrices(cm=cm, top_k_cm=15, exclude_o_in_plots=True, o_id=o_id, id2label=id2label)
    plot_f1_top_bottom(y_true_np=y_true_np, y_pred_np=y_pred_np, labels_sorted=labels_sorted, all_class_ids=np.arange(num_classes), support_counts=support_counts, exclude_o_in_plots=True, o_id=o_id, id2label=id2label, top_n_f1=12, bottom_n_f1=12)
    plot_precision_recall_f1_grouped(y_true_np=y_true_np, y_pred_np=y_pred_np, labels_sorted=labels_sorted, all_class_ids=np.arange(num_classes), support_counts=support_counts, exclude_o_in_plots=True, o_id=o_id, id2label=id2label, num_grouped_labels=10)
    print_top_confusions(cm=cm, id2label=id2label, num_classes=num_classes, top_n=15)

    plot_performance_by_sentence_length(
        raw_test_tokens=raw_test_tokens_dropped,
        raw_test_bio_labels=raw_test_bio_labels_dropped,
        pred_ids=pred_ids,
        label2id=label2id,
        o_id=o_id,
        exclude_o=False,
        metric="accuracy",
        bins=[(0, 20), (21, 40), (41, 80), (81, 100)],
        output_path="performance_by_sentence_length_accuracy.png",
    )

    plot_performance_by_sentence_length(
        raw_test_tokens=raw_test_tokens_dropped,
        raw_test_bio_labels=raw_test_bio_labels_dropped,
        pred_ids=pred_ids,
        label2id=label2id,
        o_id=o_id,
        exclude_o=False,
        metric="f1",
        bins=[(0, 20), (21, 40), (41, 80), (81, 100)],
        output_path="performance_by_sentence_length_f1.png",
    )

    plot_false_negative_false_positive_by_label(y_true_np=y_true_np, y_pred_np=y_pred_np, all_class_ids=np.arange(num_classes), support_counts=support_counts, id2label=id2label, o_id=o_id, exclude_o_in_plots=True, num_labels=10, sort_by="total", output_path="false_negative_false_positive_by_label.png")
    plot_bio_error_type_chart(y_true_np=y_true_np, y_pred_np=y_pred_np, id2label=id2label, pad_id=None, o_label="O", output_path="bio_error_type_chart.png")
    overall_acc = np.mean(y_true_np == y_pred_np)
    private_mask = (y_true_np != o_id)
    private_acc = np.mean(y_true_np[private_mask] == y_pred_np[private_mask])
    plot_accuracy_summary(overall_acc=overall_acc, private_acc=private_acc, output_path="accuracy_summary.png")
    plot_sentence_length_distribution(raw_tokens=raw_all_tokens, max_line=100, output_path="sentence_length_distribution.png")