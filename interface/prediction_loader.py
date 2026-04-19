import json
from tensorflow.keras.models import load_model
from gensim.models import Word2Vec

from config import FULL_MODEL_PATH, WV_MODEL_PATH, CHAR_VOCAB_PATH, META_PATH, LABEL_MAP_PATH
from data.data_pipeline import (
    prepare_dataset,
    build_word_index_and_embedding,
    obtain_raw_tokens_and_labels_from_raw_dataset,
    build_word_char_sequences_multiclass,
    drop_long_sentences
)


def load_prediction_artifacts():
    trained_model = load_model(FULL_MODEL_PATH, compile=False)
    trained_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

    loaded_w2v = Word2Vec.load(WV_MODEL_PATH)

    with open(CHAR_VOCAB_PATH, "r", encoding="utf-8") as f:
        char2id = json.load(f)

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label2id = json.load(f)

    id2label = {int(v): k for k, v in label2id.items()}
    num_classes = int(meta["num_classes"])
    word_index, _ = build_word_index_and_embedding(loaded_w2v)

    return {
        "trained_model": trained_model,
        "char2id": char2id,
        "meta": meta,
        "label2id": label2id,
        "id2label": id2label,
        "num_classes": num_classes,
        "word_index": word_index,
    }


def prepare_test_data(train_size, test_size, word_index, char2id, label2id):
    ds_train, ds_test = prepare_dataset(train_size, test_size)

    raw_train_tokens, raw_train_bio_labels = obtain_raw_tokens_and_labels_from_raw_dataset(ds_train)
    raw_test_tokens, raw_test_bio_labels = obtain_raw_tokens_and_labels_from_raw_dataset(ds_test)
    raw_test_tokens, raw_test_bio_labels, _ = drop_long_sentences(raw_test_tokens, raw_test_bio_labels)

    padded_X_word_test, padded_X_char_test, padded_Y_test = build_word_char_sequences_multiclass(raw_test_tokens, raw_test_bio_labels, word_index, char2id, label2id)
    padded_X_word_training, _, _ = build_word_char_sequences_multiclass(raw_train_tokens, raw_train_bio_labels, word_index, char2id, label2id)
    return {
        "padded_X_word_training": padded_X_word_training,
        "raw_test_tokens": raw_test_tokens,
        "raw_test_bio_labels": raw_test_bio_labels,
        "padded_X_word_test": padded_X_word_test,
        "padded_X_char_test": padded_X_char_test,
        "padded_Y_test": padded_Y_test,
    }