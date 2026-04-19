from keras.callbacks import EarlyStopping, ModelCheckpoint
from config import FULL_MODEL_PATH, EPOCH_SIZE, BATCH_SIZE
from models.models import build_main_model
from utils.plot_utils import save_training_outputs


def train_and_save_model(data_bundle):
    model = build_main_model(
        data_bundle["vocab_size"],
        data_bundle["emb_matrix"],
        data_bundle["char2id"],
        data_bundle["num_classes"]
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=2,
            min_delta=0.005,
            mode="min",
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            FULL_MODEL_PATH,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1
        )
    ]

    history = model.fit(
        [data_bundle["padded_X_word_train"], data_bundle["padded_X_char_train"]],
        data_bundle["padded_Y_train"],
        sample_weight=data_bundle["sample_weights"],
        validation_data=(
            [data_bundle["padded_X_word_val"], data_bundle["padded_X_char_val"]],
            data_bundle["padded_Y_val"],
            data_bundle["sample_weights_val"]
        ),
        epochs=EPOCH_SIZE,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )

    save_training_outputs(history)

    model.save(FULL_MODEL_PATH)
    print("Model has been saved")