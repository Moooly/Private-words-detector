import tensorflow as tf
import os
import sys
import numpy as np
import json
from tokenizers.pre_tokenizers import BertPreTokenizer
from gensim.models import Word2Vec
from config import MAX_CHAR_LEN, FULL_MODEL_PATH, WV_MODEL_PATH, CHAR_VOCAB_PATH, LABEL_MAP_PATH, BATCH_SIZE
from transformers import AutoTokenizer
from keras.models import load_model
from data_pipeline import prepare_test_input_from_raw_tokens, build_word_index_and_embedding
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
tf.keras.mixed_precision.set_global_policy("mixed_float16")
tf.config.set_visible_devices([], 'GPU')
from transformers import AutoTokenizer

# choose the same tokenizer you used when preparing data/training
tok = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

# custom_text = "I'm Moly, Shirley is my wife, she lives in Burnaby"
# pre = BertPreTokenizer()
# words_and_spans = pre.pre_tokenize_str(custom_text)

trained_model = load_model(FULL_MODEL_PATH)
loaded_w2v = Word2Vec.load(WV_MODEL_PATH)
word_index, _ = build_word_index_and_embedding(loaded_w2v)
# print("...............Custom Test..................")
with open(CHAR_VOCAB_PATH, "r", encoding="utf-8") as f:
    char2id = json.load(f)
with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    label2id = json.load(f)
    
id2label = {int(v): k for k, v in label2id.items()}
# # custom_text = sys.argv[1]

# raw_tokens = [hf_tokenize(custom_text)]
# max_sentence_len = len(raw_tokens[0])
# Xw, Xc = prepare_test_input_from_raw_tokens(
#     raw_tokens,
#     word_index,
#     char2id,
#     max_sentence_len=max_sentence_len,
#     max_char_len=MAX_CHAR_LEN
# )
# # Ensure dense numeric dtypes
# Xw = np.asarray(Xw, dtype=np.int32)
# Xc = np.asarray(Xc, dtype=np.int32)

# probs = trained_model.predict([Xw, Xc], batch_size=1, verbose=0)  # (1, T, C)
# pred_ids = np.argmax(probs, axis=-1)[0]                           # (T,)

# tokens = raw_tokens[0]
# T = len(tokens)

# pred_label_ids = pred_ids[:T].tolist()
# pred_labels = [id2label.get(int(i), "<UNK_LABEL>") for i in pred_label_ids]

# print("\nToken\tPred")
# for tok, lab in zip(tokens, pred_labels):
#     print(f"{tok}\t{lab}")

from tokenizers.pre_tokenizers import BertPreTokenizer

pre = BertPreTokenizer()

custom_text = "I'm Moly, I live in Burnaby, my phone number is 2508792372, my email address is immoooly@gmail.com. My wife is Shirley, she loves gaming and She lives in Kamloops"

# 1) Pre-tokenize into words (so we can align subwords back to words)
words_and_spans = pre.pre_tokenize_str(custom_text)
words = [w for (w, span) in words_and_spans]

# 2) HF tokenization with word alignment
enc = tok(
    words,
    is_split_into_words=True,
    add_special_tokens=False,
    return_attention_mask=False
)

subword_ids = enc["input_ids"]
subwords = tok.convert_ids_to_tokens(subword_ids)
word_ids = enc.word_ids()  # same length as subwords, values 0..len(words)-1

# 3) Feed subwords into YOUR existing pipeline/model
raw_tokens = [subwords]
max_sentence_len = len(subwords)

Xw, Xc = prepare_test_input_from_raw_tokens(
    raw_tokens,
    word_index,
    char2id,
    max_sentence_len=max_sentence_len,
    max_char_len=MAX_CHAR_LEN
)

Xw = np.asarray(Xw, dtype=np.int32)
Xc = np.asarray(Xc, dtype=np.int32)

probs = trained_model.predict([Xw, Xc], batch_size=BATCH_SIZE, verbose=0)[0]  # (T, C)

# 4) Subword -> word: pick the subword with the highest confidence for each word
# token_pred_id = probs.argmax(axis=-1)   # (T,)
# token_conf = probs.max(axis=-1)         # (T,)

# word_pred_ids = []
# word_conf = []

# for w_i in range(len(words)):
#     idxs = [t for t, wid in enumerate(word_ids) if wid == w_i]
#     if not idxs:
#         word_pred_ids.append(None)
#         word_conf.append(None)
#         continue

#     best_t = idxs[int(np.argmax(token_conf[idxs]))]  # most confident subword
#     word_pred_ids.append(int(token_pred_id[best_t]))
#     word_conf.append(float(token_conf[best_t]))

# word_pred_labels = [
#     id2label.get(i, "<UNK_LABEL>") if i is not None else "<NONE>"
#     for i in word_pred_ids
# ]

# print("\nWord\tPred\tConf")
# for w, lab, cf in zip(words, word_pred_labels, word_conf):
#     print(f"{w}\t{lab}\t{cf:.4f}" if cf is not None else f"{w}\t{lab}\tNA")



# 4) Subword -> word: OPTION C (BIO-aware) + threshold
# - pick strongest entity-type evidence across subwords (excluding O)
# - if passes threshold (and beats O by margin), output B- or I- based on previous word
# - otherwise output O

import numpy as np

probs = probs.astype(np.float32)  # (T, C)

# ---- tune these ----
PII_THRESHOLD = 0.60   # try 0.45~0.80 (higher => fewer false positives)
MARGIN_OVER_O = 0.05   # try 0.00~0.10 (helps reduce spikes)

# ---- find O id robustly ----
if "O" in label2id:
    o_id = int(label2id["O"])
elif "0" in label2id:
    o_id = int(label2id["0"])
else:
    o_id = 0  # fallback (only if your mapping is unusual)

def parse_bio(label: str):
    """Return (prefix, entity_type). Prefix is 'B', 'I', or 'O'."""
    if label is None:
        return ("O", None)
    label = str(label)
    if label == "O":
        return ("O", None)
    if "-" in label:
        p, t = label.split("-", 1)   # p already 'B'/'I' in your dataset
        return (p, t)
    # If somehow you have non-BIO labels, treat as a type without prefix
    return (None, label)

# Build mapping: entity_type -> {"B": class_id, "I": class_id}
type2bi = {}
for cid, lab in id2label.items():
    cid = int(cid)
    if cid == o_id:
        continue
    p, t = parse_bio(lab)
    if t is None:
        continue
    if p not in ("B", "I"):
        continue
    type2bi.setdefault(t, {})
    type2bi[t][p] = cid

entity_types = list(type2bi.keys())

word_pred_ids = []
word_conf = []

prev_type = None
prev_is_entity = False

for w_i in range(len(words)):
    idxs = [t for t, wid in enumerate(word_ids) if wid == w_i]
    if not idxs:
        word_pred_ids.append(None)
        word_conf.append(None)
        prev_type = None
        prev_is_entity = False
        continue

    # Stable O probability for this word (mean across subwords)
    o_prob = float(probs[idxs, o_id].mean())

    # Find best entity TYPE evidence across subwords:
    # score(type) = max_subword max( P(B-type), P(I-type) )
    best_type = None
    best_type_prob = -1.0

    for t in entity_types:
        b_id = type2bi[t].get("B")
        i_id = type2bi[t].get("I")
        cand_ids = [x for x in (b_id, i_id) if x is not None]
        if not cand_ids:
            continue

        p_sub = probs[idxs][:, cand_ids]  # (n_sub, 1 or 2)
        p_t = float(p_sub.max(axis=1).max())  # max over labels then subwords

        if p_t > best_type_prob:
            best_type_prob = p_t
            best_type = t

    # Threshold decision (Option C)
    if (best_type is not None) and (best_type_prob >= PII_THRESHOLD) and (best_type_prob >= o_prob + MARGIN_OVER_O):
        # Decide B vs I based on previous predicted word
        prefix = "I" if (prev_is_entity and prev_type == best_type) else "B"

        chosen_id = type2bi[best_type].get(prefix)
        if chosen_id is None:
            # fallback in case dataset lacks I-* for some type
            chosen_id = type2bi[best_type].get("B") or type2bi[best_type].get("I")

        pred_id = int(chosen_id)
        conf = float(best_type_prob)

        prev_is_entity = True
        prev_type = best_type
    else:
        pred_id = int(o_id)
        conf = float(o_prob)

        prev_is_entity = False
        prev_type = None

    word_pred_ids.append(pred_id)
    word_conf.append(conf)

word_pred_labels = [
    id2label.get(int(i), "<UNK_LABEL>") if i is not None else "<NONE>"
    for i in word_pred_ids
]

print("\nWord\tPred\tConf")
for w, lab, cf in zip(words, word_pred_labels, word_conf):
    print(f"{w}\t{lab}\t{cf:.4f}" if cf is not None else f"{w}\t{lab}\tNA")