import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score



def short_label(s: str, max_len: int = 18) -> str:
    s = str(s)
    if len(s) <= max_len:
        return s
    if "-" in s:
        p, t = s.split("-", 1)
        t = t[: max_len - (len(p) + 1)]
        return f"{p}-{t}"
    return s[:max_len]


def pretty_label(s: str) -> str:
    s = str(s)
    if "-" in s:
        p, t = s.split("-", 1)
        return f"{p}\n{t}"
    return s


def cell_text_color(im, value: float) -> str:
    rgba = im.cmap(im.norm(value))
    r, g, b = rgba[:3]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < 0.45 else "black"


def plot_confusion_matrices(cm, top_k_cm, exclude_o_in_plots, o_id, id2label):
    num_classes = cm.shape[0]
    all_class_ids = np.arange(num_classes)

    if exclude_o_in_plots:
        candidate_ids = all_class_ids[all_class_ids != o_id]
    else:
        candidate_ids = all_class_ids

    support_counts = cm.sum(axis=1)
    sorted_by_support = candidate_ids[np.argsort(support_counts[candidate_ids])[::-1]]
    top_ids = sorted_by_support[:top_k_cm]

    cm_small = cm[np.ix_(top_ids, top_ids)]
    row_sums = cm_small.sum(axis=1, keepdims=True)
    cm_small_norm = cm_small / np.maximum(row_sums, 1)

    small_names = [short_label(id2label[int(i)]) for i in top_ids]
    small_names_pretty = [pretty_label(name) for name in small_names]
    small_names_x = [name.replace("\n", "-") for name in small_names_pretty]
    y_tick_labels = small_names_x

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_small_norm, interpolation="nearest")
    ax.set_title(f"Confusion Matrix (Top {top_k_cm} by support, No PAD)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Row-normalized proportion")

    for r in range(cm_small.shape[0]):
        for c in range(cm_small.shape[1]):
            count = int(cm_small[r, c])
            ratio = float(cm_small_norm[r, c])
            txt = "" if count == 0 else f"{count}\n{ratio*100:.1f}%"
            color = cell_text_color(im, ratio)
            ax.text(c, r, txt, ha="center", va="center", fontsize=7, color=color)

    ax.set_xticks(range(len(top_ids)))
    ax.set_xticklabels(small_names_x, rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
    ax.set_yticks(range(len(top_ids)))
    ax.set_yticklabels(y_tick_labels, fontsize=8)
    ax.set_xticks(np.arange(-0.5, len(top_ids), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(top_ids), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    plt.savefig("confusion_matrix_topk_norm.png", dpi=350, bbox_inches="tight")
    plt.close()
    print("Saved: confusion_matrix_topk_norm.png")

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_small, interpolation="nearest")
    ax.set_title(f"Confusion Matrix Counts (Top {top_k_cm} by support, No PAD)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Token count")

    for r in range(cm_small.shape[0]):
        for c in range(cm_small.shape[1]):
            count = int(cm_small[r, c])
            txt = str(count) if count > 0 else ""
            color = cell_text_color(im, count)
            ax.text(c, r, txt, ha="center", va="center", fontsize=8, color=color)

    ax.set_xticks(range(len(top_ids)))
    ax.set_xticklabels(small_names_x, rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
    ax.set_yticks(range(len(top_ids)))
    ax.set_yticklabels(y_tick_labels, fontsize=8)
    ax.set_xticks(np.arange(-0.5, len(top_ids), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(top_ids), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    plt.savefig("confusion_matrix_topk_counts.png", dpi=350, bbox_inches="tight")
    plt.close()
    print("Saved: confusion_matrix_topk_counts.png")


def plot_f1_top_bottom(
    y_true_np,
    y_pred_np,
    labels_sorted,
    all_class_ids,
    support_counts,
    exclude_o_in_plots,
    o_id,
    id2label,
    top_n_f1=12,
    bottom_n_f1=12,
):
    per_class_f1 = f1_score(
        y_true_np,
        y_pred_np,
        average=None,
        labels=labels_sorted,
        zero_division=0,
    )
    f1_map = {int(cls_id): float(f1) for cls_id, f1 in zip(labels_sorted, per_class_f1)}

    if exclude_o_in_plots:
        candidate_ids = all_class_ids[all_class_ids != o_id]
    else:
        candidate_ids = all_class_ids

    candidate_f1 = np.array([f1_map[int(i)] for i in candidate_ids])
    sorted_by_f1 = candidate_ids[np.argsort(candidate_f1)]
    bottom_ids = sorted_by_f1[:bottom_n_f1]
    top_f1_ids = sorted_by_f1[-top_n_f1:]

    show_ids = np.concatenate([bottom_ids, top_f1_ids])
    show_f1 = np.array([f1_map[int(i)] for i in show_ids])
    show_names = [short_label(id2label[int(i)]) for i in show_ids]

    order = np.argsort(show_f1)
    show_f1 = show_f1[order]
    show_names = [show_names[i] for i in order]

    plt.figure(figsize=(12, 9))
    plt.barh(range(len(show_ids)), show_f1)
    plt.yticks(range(len(show_ids)), show_names, fontsize=9)
    plt.xlabel("F1")
    plt.title(f"Per-class F1 (Bottom {bottom_n_f1} + Top {top_n_f1}, No PAD)")
    plt.xlim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig("per_class_f1_top_bottom.png", dpi=350, bbox_inches="tight")
    plt.close()
    print("Saved: per_class_f1_top_bottom.png")


def plot_precision_recall_f1_grouped(
    y_true_np,
    y_pred_np,
    labels_sorted,
    all_class_ids,
    support_counts,
    exclude_o_in_plots,
    o_id,
    id2label,
    num_grouped_labels=10,
):
    per_class_precision, per_class_recall, per_class_f1_full, _ = precision_recall_fscore_support(
        y_true_np,
        y_pred_np,
        labels=labels_sorted,
        average=None,
        zero_division=0,
    )
    precision_map = {int(cls_id): float(v) for cls_id, v in zip(labels_sorted, per_class_precision)}
    recall_map = {int(cls_id): float(v) for cls_id, v in zip(labels_sorted, per_class_recall)}
    f1_map = {int(cls_id): float(v) for cls_id, v in zip(labels_sorted, per_class_f1_full)}

    if exclude_o_in_plots:
        candidate_ids = all_class_ids[all_class_ids != o_id]
    else:
        candidate_ids = all_class_ids

    sorted_grouped_ids = candidate_ids[np.argsort(support_counts[candidate_ids])[::-1]]
    group_ids = sorted_grouped_ids[:num_grouped_labels]

    group_precision = np.array([precision_map[int(i)] for i in group_ids])
    group_recall = np.array([recall_map[int(i)] for i in group_ids])
    group_f1 = np.array([f1_map[int(i)] for i in group_ids])
    group_support = support_counts[group_ids]
    group_names = [short_label(id2label[int(i)], max_len=20) for i in group_ids]

    order = np.argsort(group_f1)
    group_precision = group_precision[order]
    group_recall = group_recall[order]
    group_f1 = group_f1[order]
    group_support = group_support[order]
    group_names = [group_names[i] for i in order]

    x = np.arange(len(group_names))
    width = 0.24

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(x - width, group_precision, width, label="Precision")
    ax.bar(x, group_recall, width, label="Recall")
    ax.bar(x + width, group_f1, width, label="F1-score")

    ax.set_title(f"Precision / Recall / F1 by Label (Top {num_grouped_labels} by support, No PAD)")
    ax.set_xlabel("Label")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(group_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for i, n in enumerate(group_support):
        ax.text(i, 0.02, f"n={int(n)}", ha="center", va="bottom", fontsize=8, rotation=90)

    plt.tight_layout()
    plt.savefig("precision_recall_f1_grouped_top_labels.png", dpi=350, bbox_inches="tight")
    plt.close()
    print("Saved: precision_recall_f1_grouped_top_labels.png")


def print_top_confusions(cm, id2label, num_classes, top_n=15):
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    flat_idx = np.argsort(cm_no_diag.ravel())[::-1]

    print("\n========== Top Confusions (No PAD) ==========")
    shown = 0
    for idx in flat_idx:
        val = cm_no_diag.ravel()[idx]
        if val <= 0:
            break
        i = idx // num_classes
        j = idx % num_classes
        print(f"{id2label[i]} -> {id2label[j]}: {int(val)}")
        shown += 1
        if shown >= top_n:
            break

def plot_performance_by_sentence_length(
    raw_test_tokens,
    raw_test_bio_labels,
    pred_ids,
    label2id,
    o_id=None,
    exclude_o=False,
    metric="f1",
    bins=None,
    output_path=None,
):

    """Plot model performance grouped by sentence length."""
    if bins is None:
        bins = [(0, 20), (21, 40), (41, 60), (61, 80), (81, 100)]

    if output_path is None:
        output_path = f"performance_by_sentence_length_{metric}.png"

    bucket_names = [f"{lo}–{hi}" for lo, hi in bins]
    bucket_scores = []
    bucket_counts = []

    for lo, hi in bins:
        group_true = []
        group_pred = []
        sentence_count = 0

        for i in range(len(raw_test_tokens)):
            sent = raw_test_tokens[i]
            sent_len = len(sent)
            if not (lo <= sent_len <= hi):
                continue

            true_lbs = raw_test_bio_labels[i]
            true_ids = np.array([label2id[l] for l in true_lbs], dtype=np.int32)
            pred_seq = np.array(pred_ids[i][: len(sent)], dtype=np.int32)
            print(f"true_ids.shape: {true_ids.shape}")
            print(f"pred_seq.shape: {pred_seq.shape}")

            if exclude_o and o_id is not None:
                keep = true_ids != o_id
                true_ids = true_ids[keep]
                pred_seq = pred_seq[keep]

            if len(true_ids) == 0:
                continue

            group_true.extend(true_ids.tolist())
            group_pred.extend(pred_seq.tolist())
            sentence_count += 1
        
        print(f"len(group_true): {len(group_true)}")
        print(f"len(group_pred): {len(group_pred)}")
        bucket_counts.append(sentence_count)

        if len(group_true) == 0:
            bucket_scores.append(0.0)
        else:
            group_true = np.array(group_true, dtype=np.int32)
            group_pred = np.array(group_pred, dtype=np.int32)

            if metric == "accuracy":
                score = accuracy_score(group_true, group_pred)
            else:
                labels_present = np.unique(np.concatenate([group_true, group_pred]))
                score = f1_score(
                    group_true,
                    group_pred,
                    labels=labels_present,
                    average="macro",
                    zero_division=0,
                )
            bucket_scores.append(float(score))

    x = np.arange(len(bucket_names))
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, bucket_scores)

    ax.set_title(f"Performance by Sentence Length ({metric.upper()})")
    ax.set_xlabel("Sentence length group")
    ax.set_ylabel(metric.upper())
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_names, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for i, (bar, score, count) in enumerate(zip(bars, bucket_scores, bucket_counts)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(score + 0.02, 0.98),
            f"{score:.3f}\n(n={count})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

def plot_false_negative_false_positive_by_label(
    y_true_np,
    y_pred_np,
    all_class_ids,
    support_counts,
    id2label,
    o_id=None,
    exclude_o_in_plots=True,
    num_labels=10,
    sort_by="fn",
    output_path="false_negative_false_positive_by_label.png",
):
    """
    Plot false negatives and false positives for selected labels.

    FN for class c: true=c but pred!=c
    FP for class c: true!=c but pred=c
    """
    if exclude_o_in_plots and o_id is not None:
        candidate_ids = all_class_ids[all_class_ids != o_id]
    else:
        candidate_ids = all_class_ids

    fn_counts = []
    fp_counts = []

    for cls_id in candidate_ids:
        fn = int(np.sum((y_true_np == cls_id) & (y_pred_np != cls_id)))
        fp = int(np.sum((y_true_np != cls_id) & (y_pred_np == cls_id)))
        fn_counts.append(fn)
        fp_counts.append(fp)

    fn_counts = np.array(fn_counts, dtype=np.int32)
    fp_counts = np.array(fp_counts, dtype=np.int32)

    # choose labels with highest total error
    total_error = fn_counts + fp_counts
    selected_order = np.argsort(total_error)[::-1]
    selected_ids = candidate_ids[selected_order][:num_labels]

    selected_fn = np.array([
        int(np.sum((y_true_np == cls_id) & (y_pred_np != cls_id)))
        for cls_id in selected_ids
    ])
    selected_fp = np.array([
        int(np.sum((y_true_np != cls_id) & (y_pred_np == cls_id)))
        for cls_id in selected_ids
    ])
    selected_support = support_counts[selected_ids]
    selected_names = [short_label(id2label[int(i)], max_len=20) for i in selected_ids]

    # optional re-sorting for display
    if sort_by == "fp":
        order = np.argsort(selected_fp)[::-1]
    elif sort_by == "total":
        order = np.argsort(selected_fn + selected_fp)[::-1]
    else:  # default = fn
        order = np.argsort(selected_fn)[::-1]

    selected_fn = selected_fn[order]
    selected_fp = selected_fp[order]
    selected_support = selected_support[order]
    selected_names = [selected_names[i] for i in order]

    x = np.arange(len(selected_names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(14, 8))
    bars_fn = ax.bar(x - width / 2, selected_fn, width, label="False Negative")
    bars_fp = ax.bar(x + width / 2, selected_fp, width, label="False Positive")

    ax.set_title(f"False Negative / False Positive Comparison by Label (Top {num_labels}, No O)")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(selected_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    ymax = max(
        1,
        np.max(np.concatenate([selected_fn, selected_fp]))
    )
    ax.set_ylim(0, ymax * 1.18)

    for b1, b2, fn, fp in zip(bars_fn, bars_fp, selected_fn, selected_fp):
        ax.text(
            b1.get_x() + b1.get_width() / 2,
            b1.get_height() + ymax * 0.02,
            str(int(fn)),
            ha="center",
            va="bottom",
            fontsize=9,
        )
        ax.text(
            b2.get_x() + b2.get_width() / 2,
            b2.get_height() + ymax * 0.02,
            str(int(fp)),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

def _split_bio(label: str):
    """Return (prefix, entity_type). For O -> ('O', None)."""
    label = str(label)
    if label == "O":
        return "O", None
    if "-" in label:
        prefix, entity_type = label.split("-", 1)
        return prefix, entity_type
    return label, None


def plot_bio_error_type_chart(
    y_true_np,
    y_pred_np,
    id2label,
    pad_id=None,
    o_label="O",
    output_path="bio_error_type_chart.png",
):
    """
    Count BIO-style error categories and plot them as a bar chart.

    Error categories:
    1. True private token predicted as O
    2. Wrong entity type
    3. B- predicted as I- (same entity type)
    4. I- predicted as B- (same entity type)
    5. Other BIO errors
    """
    error_counts = {
        "Private → O": 0,
        "Wrong entity type": 0,
        "B → I": 0,
        "I → B": 0,
        "Other BIO errors": 0,
    }

    for yt, yp in zip(y_true_np, y_pred_np):
        yt = int(yt)
        yp = int(yp)

        if pad_id is not None and yt == pad_id:
            continue

        true_label = str(id2label[yt])
        pred_label = str(id2label[yp])

        if true_label == pred_label:
            continue

        true_prefix, true_type = _split_bio(true_label)
        pred_prefix, pred_type = _split_bio(pred_label)

        # true private token predicted as O
        if true_label != o_label and pred_label == o_label:
            error_counts["Private → O"] += 1

        # same entity type, but wrong BIO boundary
        elif true_prefix == "B" and pred_prefix == "I" and true_type == pred_type:
            error_counts["B → I"] += 1
        elif true_prefix == "I" and pred_prefix == "B" and true_type == pred_type:
            error_counts["I → B"] += 1

        # both are entity labels, but wrong type
        elif true_label != o_label and pred_label != o_label and true_type != pred_type:
            error_counts["Wrong entity type"] += 1

        # everything else
        else:
            error_counts["Other BIO errors"] += 1

    categories = list(error_counts.keys())
    counts = list(error_counts.values())

    x = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(x, counts)

    ax.set_title("BIO Error Type Chart")
    ax.set_xlabel("Error category")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    ymax = max(counts) if len(counts) > 0 else 0
    upper = max(1, int(ymax * 1.15))
    ax.set_ylim(0, upper)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(1, upper * 0.01),
            str(int(count)),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

def plot_accuracy_summary(overall_acc, private_acc, output_path="accuracy_summary.png"):
    labels = [
        "Overall Accuracy\n(with O)",
        "Private Token Accuracy\n(without O)",
    ]
    values = [overall_acc, private_acc]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values)

    ax.set_title("Test Accuracy Summary")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(v + 0.02, 0.98),
            f"{v:.4f}\n({v*100:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

import matplotlib.pyplot as plt


def save_custom_entity_table_figure(strict_demo_entities, output_path="custom_entity_output_table.png"):
    rows = [[str(i), entity_text, entity_label]
            for i, (entity_text, entity_label) in enumerate(strict_demo_entities, start=1)]

    fig_height = max(3.0, 0.45 * len(rows) + 1.3)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis("off")
    ax.set_title("Predicted Entities for Custom Input", fontsize=13, pad=12)

    table = ax.table(
        cellText=rows,
        colLabels=["Index", "Entity Text", "Predicted Entity Type"],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.35)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

def plot_sentence_length_distribution(
    raw_tokens,
    max_line=150,
    output_path="sentence_length_distribution.png",
):
    lengths = [len(sent) for sent in raw_tokens]
    if len(lengths) == 0:
        print("No sentence lengths available.")
        return

    total = len(lengths)
    above_count = sum(x > max_line for x in lengths)
    above_pct = 100.0 * above_count / total

    # Use a tighter x-axis focused on the main distribution.
    # Keep the max-line visible, but avoid stretching the plot because of a few extreme outliers.
    p99 = int(np.percentile(lengths, 99))
    x_max = max(max_line + 10, p99)

    clipped_lengths = [min(x, x_max) for x in lengths]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(clipped_lengths, bins=30, edgecolor="black")

    ax.axvline(
        x=max_line,
        linestyle="--",
        linewidth=2,
        label=f"Max length = {max_line}"
    )

    ax.set_xlim(0, x_max)
    ax.set_title("Sentence Length Distribution")
    ax.set_xlabel("Number of Tokens")
    ax.set_ylabel("Number of Sentences")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    ax.text(
        0.98,
        0.95,
        f"> {max_line} tokens: {above_count}/{total} ({above_pct:.2f}%)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")