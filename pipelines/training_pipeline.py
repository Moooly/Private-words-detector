import json
import os
import pickle
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from config import WV_MODEL_PATH, MAX_CHAR_LEN, CHAR_VOCAB_PATH, META_PATH, WV_META_PATH, PREPARED_DATA_PATH
from data.data_pipeline import (
    prepare_dataset,
    train_word2Vec,
    build_word_index_and_embedding,
    build_char_vocab,
    build_label_map,
    build_word_char_sequences_multiclass,
    compute_sample_weights_multiclass_balanced,
    obtain_raw_tokens_and_labels_from_raw_dataset,
    drop_long_sentences,
)


def load_prepared_data_bundle(train_size, test_size):
    if not os.path.exists(PREPARED_DATA_PATH):
        return None

    try:
        with open(PREPARED_DATA_PATH, "rb") as f:
            saved_obj = pickle.load(f)

        saved_train_size = saved_obj.get("train_size")
        saved_test_size = saved_obj.get("test_size")
        saved_bundle = saved_obj.get("bundle")

        if saved_train_size == train_size and saved_test_size == test_size:
            print("Reusing prepared data bundle from local storage")
            return saved_bundle
    except Exception as e:
        print(f"Could not load prepared data bundle: {e}")

    return None


def save_prepared_data_bundle(train_size, test_size, bundle):
    os.makedirs(os.path.dirname(PREPARED_DATA_PATH), exist_ok=True)
    with open(PREPARED_DATA_PATH, "wb") as f:
        pickle.dump(
            {
                "train_size": train_size,
                "test_size": test_size,
                "bundle": bundle,
            },
            f,
        )

def get_or_train_word2vec(raw_train_tokens, train_size, test_size):
    if os.path.exists(WV_MODEL_PATH) and os.path.exists(WV_META_PATH):
        try:
            with open(WV_META_PATH, "r", encoding="utf-8") as f:
                stored_meta = json.load(f)
            stored_train_size = stored_meta.get("train_size")
            stored_test_size = stored_meta.get("test_size")

            if stored_train_size == train_size and stored_test_size == test_size:
                print("Reusing saved Word2Vec model from storage")
                return Word2Vec.load(WV_MODEL_PATH)
        except Exception as e:
            print(f"Could not reuse saved Word2Vec model: {e}")

    print("Training new Word2Vec model...")
    w2v_model = train_word2Vec(raw_train_tokens)
    w2v_model.save(WV_MODEL_PATH)
    with open(WV_META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_size": train_size,
                "test_size": test_size,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return w2v_model

def prepare_data(train_size, test_size, use_cache=True):
    if use_cache:
        cached_bundle = load_prepared_data_bundle(train_size, test_size)
        if cached_bundle is not None:
            return cached_bundle

    train_dataset, test_dataset = prepare_dataset(train_size, test_size)
    raw_tokens_pre, raw_bio_labels = obtain_raw_tokens_and_labels_from_raw_dataset(train_dataset)
    raw_test_tokens_pre, raw_test_bio_labels_pre = obtain_raw_tokens_and_labels_from_raw_dataset(test_dataset)
    
    raw_train_tokens, raw_val_tokens, raw_train_labels, raw_val_labels = train_test_split(
        raw_tokens_pre,
        raw_bio_labels,
        test_size=0.1,
        random_state=42
    )

    raw_train_tokens, raw_train_labels, dropped_train = drop_long_sentences(raw_train_tokens, raw_train_labels)
    raw_val_tokens, raw_val_labels, dropped_val = drop_long_sentences(raw_val_tokens, raw_val_labels)
    raw_test_tokens, raw_test_bio_labels, dropped_test = drop_long_sentences(raw_test_tokens_pre, raw_test_bio_labels_pre)

    print(f"{dropped_train} sentences have been dropped in training dataset")
    print(f"{dropped_val} sentences have been dropped in validation dataset")
    print(f"{dropped_test} sentences have been dropped in test dataset")
    
    w2v_model = get_or_train_word2vec(raw_train_tokens, train_size, test_size)

    word_index, emb_matrix = build_word_index_and_embedding(w2v_model)
    vocab_size = emb_matrix.shape[0]

    char2id = build_char_vocab(raw_train_tokens)
    with open(CHAR_VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(char2id, f, ensure_ascii=False)

    label2id, _, num_classes = build_label_map(raw_train_labels)
    print("NUM_CLASSES:", num_classes)

    padded_X_word_train, padded_X_char_train, padded_Y_train = build_word_char_sequences_multiclass(raw_train_tokens, raw_train_labels, word_index, char2id, label2id)
    padded_X_word_val, padded_X_char_val, padded_Y_val = build_word_char_sequences_multiclass(raw_val_tokens, raw_val_labels, word_index, char2id, label2id)
    padded_X_word_test, padded_X_char_test, padded_Y_test = build_word_char_sequences_multiclass(raw_test_tokens, raw_test_bio_labels, word_index, char2id, label2id)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"max_char_len": int(MAX_CHAR_LEN), "num_classes": int(num_classes)},
            f
        )

    sample_weights, class_weight_vec = compute_sample_weights_multiclass_balanced(padded_X_word_train, padded_Y_train, num_classes=num_classes)

    print(
        "Class weights summary (min/median/max):",
        float(class_weight_vec.min()),
        float(np.median(class_weight_vec)),
        float(class_weight_vec.max())
    )

    val_nopad = (padded_X_word_val != 0)
    sample_weights_val = class_weight_vec[padded_Y_val].astype(np.float32)
    sample_weights_val[~val_nopad] = 0.0

    bundle = {
        "raw_test_tokens_dropped": raw_test_tokens,
        "raw_test_tokens_pre": raw_test_tokens_pre,
        "raw_all_tokens": raw_tokens_pre + raw_test_tokens_pre,
        "raw_test_bio_labels_pre": raw_test_bio_labels_pre,
        "raw_test_bio_labels_dropped": raw_test_bio_labels,
        "emb_matrix": emb_matrix,
        "vocab_size": vocab_size,
        "char2id": char2id,
        "num_classes": num_classes,
        "padded_X_word_train": padded_X_word_train,
        "padded_X_char_train": padded_X_char_train,
        "padded_Y_train": padded_Y_train,
        "padded_X_word_val": padded_X_word_val,
        "padded_X_char_val": padded_X_char_val,
        "padded_Y_val": padded_Y_val,
        "padded_X_word_test": padded_X_word_test,
        "padded_X_char_test": padded_X_char_test,
        "padded_Y_test": padded_Y_test,
        "sample_weights": sample_weights,
        "sample_weights_val": sample_weights_val,
        "label2id": label2id
    }
    save_prepared_data_bundle(train_size, test_size, bundle)
    return bundle