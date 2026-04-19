from datasets import load_dataset
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from config import EMB_DIM, MAX_CHAR_LEN, EPOCH_SIZE, MAX_LEN, WINDOW_SIZE
import numpy as np
from collections import Counter
import json

# bio_labels: [[0, B-CITY, I-CITY], [0, B-FIRSTNAME, 0], [I-FIRSTNAME, B-JOBTYPE, 0]]
# label2id: {"O": 0, "B-CITY": 1, .....}
def build_label_map(bio_labels):
    unique_labels = set()
    for seq in bio_labels:
        for l in seq:
            unique_labels.add(l)

    labels = ["O"] + sorted([l for l in unique_labels if l != "O"])  # force O=0
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    with open("outputs/label2id.json", "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    return label2id, id2label, len(labels)

def process_raw_seq_to_padded_seq(raw_seq, max_len):
    return pad_sequences(raw_seq, maxlen=max_len, padding="post", truncating="post", value=0).astype(np.int32)

def prepare_dataset(train_size, test_size, seed=42):
    ds_full = load_dataset("Isotonic/pii-masking-200k", split = "train").shuffle(seed=seed)
    total_size = len(ds_full)

    train_end = int(total_size * train_size)
    test_end = int(total_size * (train_size + test_size))
    ds_train = ds_full.select(range(0, train_end))
    ds_test = ds_full.select(range(train_end, test_end))
    return ds_train, ds_test

# tokenised_text: [["My", "name", "is", "Moly"], ["I", "live", "in", "Burnaby"], ...]
# bio_labels: [["O", "O", "O", "B-NAME"], ["O", "O", "O", "B-CITY"], ...]
def obtain_raw_tokens_and_labels_from_raw_dataset(dataset):
    tokenised_text = [row["tokenised_text"] for row in dataset]
    bio_labels = [row["bio_labels"] for row in dataset]
    return tokenised_text, bio_labels

def train_word2Vec(tokenized_text):
    w2v_model = Word2Vec(vector_size=EMB_DIM, window=WINDOW_SIZE, min_count=1, sg=0)
    w2v_model.build_vocab(tokenized_text)
    w2v_model.train(tokenized_text, total_examples=len(tokenized_text), epochs=EPOCH_SIZE)
    return w2v_model

def build_word_index_and_embedding(w2v_model):
    w2v_vocab = w2v_model.wv.key_to_index
    vocab_size = len(w2v_vocab)
    PAD = "<PAD>"
    UNK = "<UNK>"
    word_index = {PAD: 0, UNK: 1} 
    emb_matrix = np.zeros((vocab_size + 2, EMB_DIM), dtype = np.float32)
    emb_matrix[1] = w2v_model.wv.vectors.mean(axis = 0) # Average value across all word embeddings goes to UNK
    for tok, idx in w2v_vocab.items():
        word_index[tok] = idx + 2
        emb_matrix[idx + 2] = w2v_model.wv[tok]
    return word_index, emb_matrix

def build_char_vocab(tokenized_text, min_freq = 1):
    PAD = "<CPAD>"
    UNK = "<CUNK>"
    counter = Counter()
    for sentence in tokenized_text:
        for tok in sentence:
            counter.update(list(tok))
    char2id = {PAD: 0, UNK: 1}
    for ch, c in counter.items():
        if c >= min_freq and ch not in char2id:
            char2id[ch] = len(char2id)
    return char2id

# return [1, 3, 4, 5, 6, 0, 0, 0, 0, 0, ...]
def token_to_char_ids(token, char2id, max_char_len = MAX_CHAR_LEN):
    PAD = "<CPAD>"
    UNK = "<CUNK>"
    pad_id = char2id[PAD]
    unk_id = char2id[UNK]

    ids = [char2id.get(c, unk_id) for c in token[:max_char_len]]
    if len(ids) < max_char_len:
        ids.extend([pad_id] * (max_char_len - len(ids)))
    return ids

def drop_long_sentences(raw_tokens_list, raw_labels_list, max_len = MAX_LEN):
    kept_tokens, kept_labels = [], []
    dropped = 0
    for toks, labs in zip(raw_tokens_list, raw_labels_list):
        if len(toks) <= max_len:
            kept_tokens.append(toks)
            kept_labels.append(labs)
        else:
            dropped += 1
    return kept_tokens, kept_labels, dropped

def build_word_char_sequences_multiclass(tokenized_text, bio_labels, word_index, char2id, label2id):

    UNK_WORD_ID = word_index["<UNK>"]

    # word_seqs: just get id version of tokenized_text(convert tokens to IDs)
    # paded_X_word: padded version of word_seqs
    word_seqs = [[word_index.get(t, UNK_WORD_ID) for t in sent] for sent in tokenized_text]
    paded_X_word = process_raw_seq_to_padded_seq(word_seqs, MAX_LEN)

    # paaded_X_char, 3-dimension matrix
    padded_X_char = np.zeros((len(tokenized_text), MAX_LEN, MAX_CHAR_LEN), dtype=np.int32)
    for i, sent in enumerate(tokenized_text):
        for j, token in enumerate(sent[:MAX_LEN]):
            padded_X_char[i, j] = token_to_char_ids(token, char2id)

    # Label ids
    y_seqs = [[label2id[l] for l in seq] for seq in bio_labels]
    padded_Y = process_raw_seq_to_padded_seq(y_seqs, MAX_LEN)

    return paded_X_word, padded_X_char, padded_Y

def compute_sample_weights_multiclass_balanced(X_word, Y, num_classes):
    X_word = np.asarray(X_word)
    Y = np.asarray(Y)
    assert X_word.shape == Y.shape

    nonpad = (X_word != 0)
    y_nonpad = Y[nonpad]

    counts = np.bincount(y_nonpad, minlength=num_classes).astype(np.float64)
    N = counts.sum()
    K = float(num_classes)

    class_weight = np.zeros(num_classes, dtype=np.float32)
    for k in range(num_classes):
        ck = counts[k]
        class_weight[k] = float(N / (K * ck)) if ck > 0 else 0.0
    class_weight = np.clip(class_weight, 0.0, 50.0)

    # replace each class ID in Y with its corresponding weight
    # class_weight = [0.2, 4.0, 6.0, 8.0]
    # Y = [[0, 0, 3, 1], [0, 2, 0, 0]]
    # sample_weight = [[0.2, 0.2, 8.0, 4.0], [0.2, 6.0, 0.2, 0.2]]
    sample_weight = class_weight[Y].astype(np.float32) 
    sample_weight[~nonpad] = 0.0
    return sample_weight, class_weight

def prepare_test_input_from_raw_tokens(list_of_token_list, word_index, char2id, max_sentence_len=None, max_char_len = MAX_CHAR_LEN):
    UNK_WORD_ID = word_index["<UNK>"]
    word_seqs = [[word_index.get(token, UNK_WORD_ID) for token in sentence] for sentence in list_of_token_list]
    max_len = 0
    if max_sentence_len is None:
        max_len = max(len(s) for s in word_seqs) if word_seqs else 0
    else:
        max_len = max_sentence_len

    X_word = pad_sequences(word_seqs, maxlen = max_len, padding = "post", value = 0)
    X_char = np.zeros((len(list_of_token_list), max_len, max_char_len), dtype=np.int32)
    for i, sentence in enumerate(list_of_token_list):
        for j, token in enumerate(sentence[:max_len]):
            X_char[i, j] = token_to_char_ids(token, char2id, max_char_len=max_char_len)
    return X_word, X_char