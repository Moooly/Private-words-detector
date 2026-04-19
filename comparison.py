import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report

from config import NUM_OF_CONV_LAYERS, DROPOUT_RATE, KEY_DIM, MAX_CHAR_LEN, CHAR_DIM, CHAR_CNN_FILTERS, EPOCH_SIZE, BATCH_SIZE, FULL_MODEL_PATH, COMPARISON_CSV_PATH, COMPARISON_PNG_PATH, WITHOUT_CHAR_MODEL_PATH, WITHOUT_ATTENTION_MODEL_PATH
from pipelines.training_pipeline import prepare_data
from utils.runtime_utils import configure_runtime



def build_model_variant(vocab_size, emb_matrix, char_vocab_size, num_classes, use_char=True, use_attention=True):
    word_ids = layers.Input(shape=(None,), dtype="int32", name="word_ids")

    word_x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=emb_matrix.shape[1],
        weights=[emb_matrix],
        trainable=True,
        mask_zero=False,
        name="word_emb"
    )(word_ids)

    inputs = [word_ids]
    features = [word_x]

    if use_char:
        char_ids = layers.Input(shape=(None, MAX_CHAR_LEN), dtype="int32", name="char_ids")
        inputs.append(char_ids)

        char_emb = layers.TimeDistributed(
            layers.Embedding(input_dim=char_vocab_size, output_dim=CHAR_DIM, name="char_emb"),
            name="td_char_emb"
        )(char_ids)

        char_conv_k3 = layers.TimeDistributed(
            keras.Sequential([
                layers.Conv1D(CHAR_CNN_FILTERS, kernel_size=3, padding="same", activation="relu"),
                layers.GlobalMaxPooling1D(),
            ]),
            name="td_char_encoder_k3"
        )(char_emb)

        char_feat = char_conv_k3
        features.append(char_feat)

    if len(features) > 1:
        x = layers.Concatenate(name="concat_word_char")(features)
    else:
        x = features[0]

    x = layers.Conv1D(NUM_OF_CONV_LAYERS, 3, padding="same", activation="relu", name="token_cnn")(x)
    x = layers.Dropout(DROPOUT_RATE, name="dropout_after_cnn")(x)

    if use_attention:
        mask = tf.not_equal(word_ids, 0)
        attn_mask = tf.expand_dims(tf.expand_dims(mask, axis=1), axis=1)

        attn_layer = layers.MultiHeadAttention(num_heads=2, key_dim=KEY_DIM, name="mha")
        attn_out = attn_layer(query=x, value=x, key=x, attention_mask=attn_mask)

        x = layers.LayerNormalization(name="attn_residual_norm")(x + attn_out)
        x = layers.Dropout(DROPOUT_RATE, name="dropout_after_attn")(x)

    logits = layers.Dense(num_classes, activation="softmax", name="tag_probs")(x)

    model = keras.Model(inputs, logits)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model

def evaluate_token_metrics(model, X_word, X_char, Y_true, label2id, batch_size=32):
    if len(model.inputs) == 2:
        probs = model.predict([X_word, X_char], batch_size=batch_size, verbose=0)
    else:
        probs = model.predict(X_word, batch_size=batch_size, verbose=0)

    y_pred = np.argmax(probs, axis=-1)

    pad_mask = (X_word != 0)
    y_true_flat = Y_true[pad_mask]
    y_pred_flat = y_pred[pad_mask]

    overall_acc = accuracy_score(y_true_flat, y_pred_flat)

    o_id = label2id.get("O", None)
    if o_id is not None:
        private_mask = (y_true_flat != o_id)
        if np.sum(private_mask) > 0:
            private_acc = accuracy_score(y_true_flat[private_mask], y_pred_flat[private_mask])
        else:
            private_acc = 0.0
    else:
        private_acc = 0.0

    labels_present = np.unique(np.concatenate([y_true_flat, y_pred_flat]))
    macro_f1 = f1_score(
        y_true_flat,
        y_pred_flat,
        labels=labels_present,
        average="macro",
        zero_division=0,
    )

    return {
        "overall_accuracy": float(overall_acc),
        "private_accuracy_wo_O": float(private_acc),
        "macro_f1": float(macro_f1),
    }


# OOV token accuracy evaluation
def evaluate_oov_token_metrics(model, X_word, X_char, Y_true, label2id, unk_word_id=1, batch_size=32):
    if len(model.inputs) == 2:
        probs = model.predict([X_word, X_char], batch_size=batch_size, verbose=0)
    else:
        probs = model.predict(X_word, batch_size=batch_size, verbose=0)

    y_pred = np.argmax(probs, axis=-1)

    oov_mask = (X_word == unk_word_id)

    y_true_oov = Y_true[oov_mask]
    y_pred_oov = y_pred[oov_mask]

    if len(y_true_oov) == 0:
        return {
            "oov_accuracy": 0.0,
            "oov_count": 0,
            "oov_accuracy_wo_O": 0.0,
            "oov_macro_f1_wo_O": 0.0,
            "oov_private_count": 0,
        }

    oov_acc = accuracy_score(y_true_oov, y_pred_oov)

    o_id = label2id.get("O", None)
    if o_id is not None:
        private_oov_mask = (y_true_oov != o_id)
        y_true_oov_private = y_true_oov[private_oov_mask]
        y_pred_oov_private = y_pred_oov[private_oov_mask]
    else:
        y_true_oov_private = y_true_oov
        y_pred_oov_private = y_pred_oov

    if len(y_true_oov_private) == 0:
        oov_acc_wo_O = 0.0
        oov_macro_f1_wo_O = 0.0
        oov_private_count = 0
    else:
        oov_acc_wo_O = accuracy_score(y_true_oov_private, y_pred_oov_private)
        labels_present = np.unique(np.concatenate([y_true_oov_private, y_pred_oov_private]))
        oov_macro_f1_wo_O = f1_score(
            y_true_oov_private,
            y_pred_oov_private,
            labels=labels_present,
            average="macro",
            zero_division=0,
        )
        oov_private_count = int(len(y_true_oov_private))

    return {
        "oov_accuracy": float(oov_acc),
        "oov_count": int(len(y_true_oov)),
        "oov_accuracy_wo_O": float(oov_acc_wo_O),
        "oov_macro_f1_wo_O": float(oov_macro_f1_wo_O),
        "oov_private_count": int(oov_private_count),
    }


def train_and_compare_variant(name, use_char, use_attention, save_model_path):
    
    train_size = 0.7
    test_size = 0.3
    data_bundle = prepare_data(train_size, test_size)

    X_word_train=data_bundle["padded_X_word_train"]
    X_char_train=data_bundle["padded_X_char_train"]
    Y_train=data_bundle["padded_Y_train"]
    sample_weights_train=data_bundle["sample_weights"]
    X_word_val=data_bundle["padded_X_word_val"]
    X_char_val=data_bundle["padded_X_char_val"]
    Y_val=data_bundle["padded_Y_val"]
    sample_weights_val=data_bundle["sample_weights_val"]
    X_word_test=data_bundle["padded_X_word_test"]
    X_char_test=data_bundle["padded_X_char_test"]
    Y_test=data_bundle["padded_Y_test"]
    vocab_size=data_bundle["vocab_size"]
    emb_matrix=data_bundle["emb_matrix"]
    char_vocab_size=len(data_bundle["char2id"])
    num_classes=data_bundle["num_classes"]
    label2id=data_bundle["label2id"]

    if os.path.exists(save_model_path):
        print(f"Loading stored variant model from {save_model_path}")
        model = keras.models.load_model(save_model_path)
    else:
        model = build_model_variant(
            vocab_size=vocab_size,
            emb_matrix=emb_matrix,
            char_vocab_size=char_vocab_size,
            num_classes=num_classes,
            use_char=use_char,
            use_attention=use_attention,
        )

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=2,
                min_delta=0.005,
                mode="min",
                restore_best_weights=True,
                verbose=1,
            ),
            ModelCheckpoint(
                save_model_path,
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                verbose=1,
            ),
        ]

        if use_char:
            train_inputs = [X_word_train, X_char_train]
            val_inputs = ([X_word_val, X_char_val], Y_val, sample_weights_val)
        else:
            train_inputs = X_word_train
            val_inputs = (X_word_val, Y_val, sample_weights_val)

        model.fit(
            train_inputs,
            Y_train,
            sample_weight=sample_weights_train,
            validation_data=val_inputs,
            epochs=EPOCH_SIZE,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
        )

    metrics = evaluate_token_metrics(
        model=model,
        X_word=X_word_test,
        X_char=X_char_test,
        Y_true=Y_test,
        label2id=label2id,
        batch_size=BATCH_SIZE,
    )

    result = {
        "Model": name,
        "Overall Accuracy": round(metrics["overall_accuracy"], 4),
        "Private Accuracy (without O)": round(metrics["private_accuracy_wo_O"], 4),
        "Macro F1": round(metrics["macro_f1"], 4),
    }
    print(result)

    if use_attention == True:
        result_of_full_model = evaluate_stored_full_model(X_word_test, X_char_test, Y_test, label2id)
        return [result_of_full_model, result]
    else:
        return [result]

def evaluate_stored_full_model(
    X_word_test,
    X_char_test,
    Y_test,
    label2id,
):
    print("\n========== Running variant: Full Model ==========")
    model = keras.models.load_model(FULL_MODEL_PATH)

    metrics = evaluate_token_metrics(
        model=model,
        X_word=X_word_test,
        X_char=X_char_test,
        Y_true=Y_test,
        label2id=label2id,
        batch_size=BATCH_SIZE,
    )

    result = {
        "Model": "Full Model",
        "Overall Accuracy": round(metrics["overall_accuracy"], 4),
        "Private Accuracy (without O)": round(metrics["private_accuracy_wo_O"], 4),
        "Macro F1": round(metrics["macro_f1"], 4),
    }
    print(result)
    return result


def plot_model_comparison(results_df, output_path=COMPARISON_PNG_PATH):
    metric_names = ["Overall Accuracy", "Private Accuracy (without O)", "Macro F1"]
    x = np.arange(len(metric_names))
    width = 0.24

    plt.figure(figsize=(10, 6))
    for idx, (_, row) in enumerate(results_df.iterrows()):
        values = [row[m] for m in metric_names]
        positions = x + (idx - 1) * width
        plt.bar(positions, values, width=width, label=row["Model"])

        for pos, val in zip(positions, values):
            plt.text(pos, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.xticks(x, metric_names)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Score")
    plt.title("Comparison of Three Model Variants")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def compare_oov_accuracy_between_models(X_word_test, X_char_test, Y_test, label2id):
    print("\n========== OOV Comparison ==========")

    full_model = keras.models.load_model(FULL_MODEL_PATH)
    without_char_model = keras.models.load_model(WITHOUT_CHAR_MODEL_PATH)

    full_oov_metrics = evaluate_oov_token_metrics(
        model=full_model,
        X_word=X_word_test,
        X_char=X_char_test,
        Y_true=Y_test,
        label2id=label2id,
        unk_word_id=1,
        batch_size=BATCH_SIZE,
    )

    without_char_oov_metrics = evaluate_oov_token_metrics(
        model=without_char_model,
        X_word=X_word_test,
        X_char=X_char_test,
        Y_true=Y_test,
        label2id=label2id,
        unk_word_id=1,
        batch_size=BATCH_SIZE,
    )

    oov_results_df = pd.DataFrame([
        {
            "Model": "Full Model",
            "OOV Accuracy": round(full_oov_metrics["oov_accuracy"], 4),
            "OOV Token Count": full_oov_metrics["oov_count"],
            "OOV Accuracy (without O)": round(full_oov_metrics["oov_accuracy_wo_O"], 4),
            "OOV Macro F1 (without O)": round(full_oov_metrics["oov_macro_f1_wo_O"], 4),
            "OOV Private Token Count": full_oov_metrics["oov_private_count"],
        },
        {
            "Model": "Without Char CNN",
            "OOV Accuracy": round(without_char_oov_metrics["oov_accuracy"], 4),
            "OOV Token Count": without_char_oov_metrics["oov_count"],
            "OOV Accuracy (without O)": round(without_char_oov_metrics["oov_accuracy_wo_O"], 4),
            "OOV Macro F1 (without O)": round(without_char_oov_metrics["oov_macro_f1_wo_O"], 4),
            "OOV Private Token Count": without_char_oov_metrics["oov_private_count"],
        },
    ])

    print(oov_results_df)
    return oov_results_df


# Private token accuracy evaluation between models
def compare_private_token_accuracy_between_models(X_word_test, X_char_test, Y_test, label2id):
    print("\n========== Private Token Comparison ==========")

    full_model = keras.models.load_model(FULL_MODEL_PATH)
    without_char_model = keras.models.load_model(WITHOUT_CHAR_MODEL_PATH)

    o_id = label2id.get("O", None)
    if o_id is None:
        raise ValueError("Label map does not contain 'O', so private-token comparison cannot be computed.")

    full_metrics = evaluate_token_metrics(
        model=full_model,
        X_word=X_word_test,
        X_char=X_char_test,
        Y_true=Y_test,
        label2id=label2id,
        batch_size=BATCH_SIZE,
    )

    without_char_metrics = evaluate_token_metrics(
        model=without_char_model,
        X_word=X_word_test,
        X_char=X_char_test,
        Y_true=Y_test,
        label2id=label2id,
        batch_size=BATCH_SIZE,
    )

    private_token_count = int(np.sum(Y_test != o_id))

    private_results_df = pd.DataFrame([
        {
            "Model": "Full Model",
            "Private Accuracy (without O)": round(full_metrics["private_accuracy_wo_O"], 4),
            "Private Token Count": private_token_count,
        },
        {
            "Model": "Without Char CNN",
            "Private Accuracy (without O)": round(without_char_metrics["private_accuracy_wo_O"], 4),
            "Private Token Count": private_token_count,
        },
    ])

    print(private_results_df)
    return private_results_df


def plot_oov_comparison(oov_results_df, output_path="outputs/oov_comparison.png"):
    metric_names = [
        "OOV Accuracy",
        "OOV Accuracy (without O)",
        "OOV Macro F1 (without O)",
    ]
    x = np.arange(len(metric_names))
    width = 0.35

    plt.figure(figsize=(9, 6))

    for idx, (_, row) in enumerate(oov_results_df.iterrows()):
        values = [row[m] for m in metric_names]
        positions = x + (idx - 0.5) * width
        plt.bar(positions, values, width=width, label=row["Model"])

        for pos, val in zip(positions, values):
            plt.text(pos, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.xticks(x, metric_names)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Score")
    plt.title("OOV Comparison Between Full Model and Without Char CNN")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    if "OOV Token Count" in oov_results_df.columns and "OOV Private Token Count" in oov_results_df.columns:
        total_oov = int(oov_results_df["OOV Token Count"].iloc[0])
        private_oov = int(oov_results_df["OOV Private Token Count"].iloc[0])
        plt.figtext(
            0.5,
            0.01,
            f"Total OOV tokens: {total_oov}    |    OOV private tokens: {private_oov}",
            ha="center",
            fontsize=10,
        )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# Plotting private token comparison
def plot_private_token_comparison(private_results_df, output_path="outputs/private_token_comparison.png"):
    plt.figure(figsize=(7, 5))
    x = np.arange(len(private_results_df))
    values = private_results_df["Private Accuracy (without O)"].values

    plt.bar(x, values, width=0.5)
    for i, val in enumerate(values):
        plt.text(i, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.xticks(x, private_results_df["Model"].values)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Private Accuracy (without O)")
    plt.title("Comparison on All Private Tokens in Test Dataset")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    if "Private Token Count" in private_results_df.columns:
        private_count = int(private_results_df["Private Token Count"].iloc[0])
        plt.figtext(
            0.5,
            0.01,
            f"Total private tokens in test dataset: {private_count}",
            ha="center",
            fontsize=10,
        )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    # configure_runtime()

    # results = []
    # results.extend(train_and_compare_variant(name="Without Char CNN", use_char=False, use_attention=True, save_model_path=WITHOUT_CHAR_MODEL_PATH))
    # results.extend(train_and_compare_variant(name="Without Attention", use_char=True, use_attention=False, save_model_path=WITHOUT_ATTENTION_MODEL_PATH))

    # results_df = pd.DataFrame(results)
    # print("\n========== Final Comparison ==========")
    # print(results_df)

    # results_df.to_csv(COMPARISON_CSV_PATH, index=False)
    # print(f"Saved: {COMPARISON_CSV_PATH}")

    # plot_model_comparison(results_df, COMPARISON_PNG_PATH)

    data_bundle = prepare_data(0.7, 0.3)
    oov_results_df = compare_oov_accuracy_between_models(
        X_word_test=data_bundle["padded_X_word_test"],
        X_char_test=data_bundle["padded_X_char_test"],
        Y_test=data_bundle["padded_Y_test"],
        label2id=data_bundle["label2id"],
    )
    oov_results_df.to_csv("outputs/oov_comparison.csv", index=False)
    print("Saved: outputs/oov_comparison.csv")
    plot_oov_comparison(oov_results_df, "outputs/oov_comparison.png")

    private_results_df = compare_private_token_accuracy_between_models(
        X_word_test=data_bundle["padded_X_word_test"],
        X_char_test=data_bundle["padded_X_char_test"],
        Y_test=data_bundle["padded_Y_test"],
        label2id=data_bundle["label2id"],
    )
    private_results_df.to_csv("outputs/private_token_comparison.csv", index=False)
    print("Saved: outputs/private_token_comparison.csv")
    plot_private_token_comparison(private_results_df, "outputs/private_token_comparison.png")


if __name__ == "__main__":
    main()