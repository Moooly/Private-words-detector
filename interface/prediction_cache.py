import os
import numpy as np
from config import BATCH_SIZE, PRED_CACHE_PATH

def build_prediction_cache_key(num_of_sentence_in_training_dataset, num_of_sentence_in_test_dataset):
    return {
        "num_of_sentence_in_training_dataset": int(num_of_sentence_in_training_dataset),
        "num_of_sentence_in_test_dataset": int(num_of_sentence_in_test_dataset),
    }

def save_cached_predictions(cache_path, cache_key, pred_ids):
    np.savez_compressed(
        cache_path,
        cache_key=cache_key,
        pred_ids=pred_ids,
    )
    print(f"Saved cached predictions to {cache_path}")


def load_cached_predictions(cache_path, cache_key):
    if not os.path.exists(cache_path):
        return None

    data = np.load(cache_path, allow_pickle=True)
    saved_key = data["cache_key"].item()

    if saved_key == cache_key:
        print(f"Loaded cached predictions from {cache_path}")
        return data["pred_ids"]

    print("Cache found but does not match current test run. Recomputing predictions...")
    return None


def get_or_compute_predictions(trained_model, padded_X_word_training, padded_X_word_test, padded_X_char_test):
    cache_key = build_prediction_cache_key(len(padded_X_word_training), len(padded_X_word_test))

    pred_ids = load_cached_predictions(PRED_CACHE_PATH, cache_key)
    if pred_ids is not None:
        return pred_ids

    probs = trained_model.predict(
        [padded_X_word_test, padded_X_char_test],
        batch_size=BATCH_SIZE,
        verbose=1,
    )
    pred_ids = np.argmax(probs, axis=-1).astype(np.int32)

    save_cached_predictions(PRED_CACHE_PATH, cache_key, pred_ids)
    return pred_ids