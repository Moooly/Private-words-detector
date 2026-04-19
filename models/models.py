import keras
from keras import layers
import tensorflow as tf
from config import NUM_OF_CONV_LAYERS, DROPOUT_RATE, KEY_DIM, MAX_CHAR_LEN, CHAR_DIM, CHAR_CNN_FILTERS


def build_main_model(vocab_size, emb_matrix, char2id, num_classes):
    word_ids = layers.Input(shape=(None,), dtype="int32", name="word_ids")
    char_ids = layers.Input(shape=(None, MAX_CHAR_LEN), dtype="int32", name="char_ids")


    word_x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=emb_matrix.shape[1],
        weights=[emb_matrix],
        trainable=True,
        mask_zero=False,
        name="word_emb"
    )(word_ids)

    char_emb = layers.TimeDistributed(
        layers.Embedding(input_dim=len(char2id), output_dim=CHAR_DIM, name="char_emb"),
        name="td_char_emb"
    )(char_ids) # output: (batch_size, sequence_length, MAX_CHAR_LEN, CHAR_DIM)

    char_conv_k3 = layers.TimeDistributed(
        keras.Sequential([
            layers.Conv1D(CHAR_CNN_FILTERS, kernel_size=3, padding="same", activation="relu"),
            layers.GlobalMaxPooling1D(),
        ]),
        name="td_char_encoder_k3"
    )(char_emb)

    x = layers.Concatenate(name="concat_word_char")([word_x, char_conv_k3])
    x = layers.Conv1D(NUM_OF_CONV_LAYERS, 3, padding="same", activation="relu")(x)
    x = layers.Dropout(DROPOUT_RATE)(x)

    mask = tf.not_equal(word_ids, 0)
    attn_mask = tf.expand_dims(tf.expand_dims(mask, axis=1), axis=1)

    attn_layer = layers.MultiHeadAttention(num_heads=2, key_dim=KEY_DIM, name="mha")
    attn_out = attn_layer(query=x, value=x, key=x, attention_mask=attn_mask)

    x = layers.LayerNormalization()(x + attn_out)
    x = layers.Dropout(DROPOUT_RATE)(x)

    logits = layers.Dense(num_classes, activation="softmax", name="tag_probs")(x)

    model = keras.Model([word_ids, char_ids], logits)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model