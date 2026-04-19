from config import MAX_CHAR_LEN
from interface.prediction_loader import load_prediction_artifacts
from interface.prediction_cache import get_or_compute_predictions
from interface.prediction_evaluation import evaluate_test_predictions, run_visualizations
from interface.custom_prediction import run_custom_demo
from pipelines.training_pipeline import prepare_data
from datasets import load_dataset
import matplotlib.pyplot as plt


def main():

    train_size = 0.7
    test_size = 0.3
    
    artifacts = load_prediction_artifacts()
    test_bundle = prepare_data(train_size, test_size)

    print(".................Start Testing..................")

    pred_ids = get_or_compute_predictions(
        trained_model=artifacts["trained_model"],
        padded_X_word_training=test_bundle["padded_X_word_train"],
        padded_X_word_test=test_bundle["padded_X_word_test"],
        padded_X_char_test=test_bundle["padded_X_char_test"]
    )

    eval_bundle = evaluate_test_predictions(
        padded_X_word_test=test_bundle["padded_X_word_test"],
        padded_Y_test=test_bundle["padded_Y_test"],
        pred_ids=pred_ids,
        label2id=artifacts["label2id"],
        num_classes=artifacts["num_classes"],
    )

    run_visualizations(
        raw_all_tokens=test_bundle["raw_all_tokens"],
        raw_test_tokens_dropped=test_bundle["raw_test_tokens_dropped"],
        raw_test_bio_labels_dropped=test_bundle["raw_test_bio_labels_dropped"],
        pred_ids=pred_ids,
        label2id=artifacts["label2id"],
        id2label=artifacts["id2label"],
        num_classes=artifacts["num_classes"],
        eval_bundle=eval_bundle,
    )

    run_custom_demo(
        trained_model=artifacts["trained_model"],
        word_index=artifacts["word_index"],
        char2id=artifacts["char2id"],
        id2label=artifacts["id2label"],
        max_char_len=MAX_CHAR_LEN,
        o_id=eval_bundle["o_id"],
    )


if __name__ == "__main__":
    main()