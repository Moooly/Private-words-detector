import re
import numpy as np

from utils.visualization import save_custom_entity_table_figure


CUSTOM_NON_O_THRESHOLD = 0.55
CUSTOM_O_MARGIN = 0.1


def build_custom_inputs_from_bert_tokens(tokens, word_index, char2id, max_char_len):
    unk_word_id = word_index.get("<UNK>", 1)
    unk_char_id = char2id.get("<UNK>", 1) if "<UNK>" in char2id else 1

    word_ids = []
    char_ids = []

    for tok in tokens:
        tok_lower = tok.lower()
        word_ids.append(word_index.get(tok_lower, unk_word_id))

        cur_char_ids = [char2id.get(ch, unk_char_id) for ch in tok[:max_char_len]]
        if len(cur_char_ids) < max_char_len:
            cur_char_ids += [0] * (max_char_len - len(cur_char_ids))
        char_ids.append(cur_char_ids)

    X_word = np.array([word_ids], dtype=np.int32)
    X_char = np.array([char_ids], dtype=np.int32)
    return X_word, X_char


def aggregate_bert_subtokens_to_words(tokens, probs, id2label, o_id, non_o_threshold=0.55, o_margin=0.08):
    merged_tokens = []
    merged_labels = []
    merged_confidences = []

    current_token_pieces = []
    current_prob_pieces = []

    def flush_current_word():
        nonlocal current_token_pieces, current_prob_pieces
        if not current_token_pieces:
            return

        word_text = "".join(current_token_pieces)
        mean_prob = np.mean(np.stack(current_prob_pieces, axis=0), axis=0)

        o_prob = float(mean_prob[o_id])
        non_o_probs = mean_prob.copy()
        non_o_probs[o_id] = -1.0
        best_non_o_id = int(np.argmax(non_o_probs))
        best_non_o_prob = float(mean_prob[best_non_o_id])

        if best_non_o_prob >= non_o_threshold and best_non_o_prob > o_prob + o_margin:
            final_label_id = best_non_o_id
            final_confidence = best_non_o_prob
        else:
            final_label_id = o_id
            final_confidence = o_prob

        final_label = id2label[final_label_id]

        merged_tokens.append(word_text)
        merged_labels.append(final_label)
        merged_confidences.append(final_confidence)

        current_token_pieces = []
        current_prob_pieces = []

    for tok, prob_vec in zip(tokens, probs):
        if tok in ["[CLS]", "[SEP]", "[PAD]"]:
            flush_current_word()
            continue

        if tok.startswith("##"):
            if current_token_pieces:
                current_token_pieces.append(tok[2:])
                current_prob_pieces.append(prob_vec)
            else:
                current_token_pieces = [tok[2:]]
                current_prob_pieces = [prob_vec]
        else:
            flush_current_word()
            current_token_pieces = [tok]
            current_prob_pieces = [prob_vec]

    flush_current_word()
    return merged_tokens, merged_labels, merged_confidences


def base_entity_type_from_label(label):
    if label == "O" or "-" not in label:
        return None
    return label.split("-", 1)[1]


def normalized_entity_type(label):
    base = base_entity_type_from_label(label)
    if base is None:
        return None

    if base in {"FIRSTNAME", "MIDDLENAME", "LASTNAME"}:
        return "PERSON_NAME"
    if base in {"CITY", "STATE", "COUNTRY"}:
        return "LOCATION"
    if base in {"EMAIL"}:
        return "EMAIL"
    if base in {"URL"}:
        return "URL"
    if base in {"PHONENUMBER", "MASKEDNUMBER"}:
        return "PHONENUMBER"
    return base


def extract_strict_entities_from_merged_labels(merged_tokens, merged_labels):
    entities = []
    i = 0
    n = len(merged_tokens)

    while i < n:
        tok_str = str(merged_tokens[i])
        ent_type = normalized_entity_type(merged_labels[i])

        if tok_str == "+" and i + 1 < n and str(merged_tokens[i + 1]).isdigit():
            entities.append((tok_str + str(merged_tokens[i + 1]), "PHONENUMBER"))
            i += 2
            continue
        if tok_str.isdigit() and len(tok_str) >= 7:
            entities.append((tok_str, "PHONENUMBER"))
            i += 1
            continue

        if i + 4 < n:
            t0 = str(merged_tokens[i])
            t1 = str(merged_tokens[i + 1])
            t2 = str(merged_tokens[i + 2])
            t3 = str(merged_tokens[i + 3])
            t4 = str(merged_tokens[i + 4])
            if t1 == "@" and t3 == "." and t0 not in {",", ".", ":"}:
                entities.append((f"{t0}@{t2}.{t4}", "EMAIL"))
                i += 5
                continue
            if t0.lower() == "www" and t1 == "." and t3 == ".":
                entities.append((f"www.{t2}.{t4}", "URL"))
                i += 5
                continue

        if ent_type is None:
            i += 1
            continue

        parts = [tok_str]
        j = i + 1
        while j < n:
            next_tok = str(merged_tokens[j])
            next_type = normalized_entity_type(merged_labels[j])
            if next_type != ent_type:
                break
            if next_tok in {",", ".", ":"}:
                break
            if ent_type not in {"PERSON_NAME", "LOCATION", "EMAIL", "URL", "PHONENUMBER"}:
                break
            parts.append(next_tok)
            j += 1

        entities.append((" ".join(parts), ent_type))
        i = j

    return entities


def build_masked_text_from_strict_entities(custom_text, strict_demo_entities):
    original_text = str(custom_text)

    entity_map = {}
    for entity_text, entity_label in strict_demo_entities:
        entity_text = str(entity_text)
        if entity_text not in entity_map:
            entity_map[entity_text] = str(entity_label)

    if not entity_map:
        return original_text

    entity_texts = sorted(entity_map.keys(), key=len, reverse=True)

    def entity_pattern(entity_text):
        escaped = re.escape(entity_text)
        starts_word = entity_text[:1].isalnum()
        ends_word = entity_text[-1:].isalnum()

        if starts_word and ends_word:
            return rf"(?<!\w){escaped}(?!\w)"
        if starts_word:
            return rf"(?<!\w){escaped}"
        if ends_word:
            return rf"{escaped}(?!\w)"
        return escaped

    pattern = "|".join(entity_pattern(t) for t in entity_texts)
    compiled = re.compile(pattern)

    def repl(match):
        matched_text = match.group(0)
        return f"[{entity_map[matched_text]}]"

    return compiled.sub(repl, original_text)


def predict_custom_sentence_with_bert_tokenizer(
    text,
    trained_model,
    bert_tokenizer,
    word_index,
    char2id,
    id2label,
    max_char_len,
    o_id,
    non_o_threshold=0.55,
    o_margin=0.08,
):
    encoded = bert_tokenizer(
        text,
        add_special_tokens=True,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    tokens = bert_tokenizer.convert_ids_to_tokens(encoded["input_ids"])

    X_word, X_char = build_custom_inputs_from_bert_tokens(
        tokens=tokens,
        word_index=word_index,
        char2id=char2id,
        max_char_len=max_char_len,
    )

    probs = trained_model.predict([X_word, X_char], verbose=0)[0]
    pred_ids_custom = np.argmax(probs, axis=-1)
    pred_labels = [id2label[int(i)] for i in pred_ids_custom[:len(tokens)]]

    merged_tokens, merged_labels, merged_confidences = aggregate_bert_subtokens_to_words(
        tokens=tokens,
        probs=probs[:len(tokens)],
        id2label=id2label,
        o_id=o_id,
        non_o_threshold=non_o_threshold,
        o_margin=o_margin,
    )
    return tokens, pred_labels, merged_tokens, merged_labels, merged_confidences


def run_custom_demo(trained_model, bert_tokenizer, word_index, char2id, id2label, max_char_len, o_id):
    custom_text = (
        "My name is Peter Parker, my phone number is +12509871234. "
        "I live in New York, my girlfriend is Mary Jane. "
        "Please feel free to ask me questions, my email is spider@gmail.com."
    )

    bert_tokens, bert_pred_labels, merged_tokens, merged_labels, merged_confidences = predict_custom_sentence_with_bert_tokenizer(
        text=custom_text,
        trained_model=trained_model,
        bert_tokenizer=bert_tokenizer,
        word_index=word_index,
        char2id=char2id,
        id2label=id2label,
        max_char_len=max_char_len,
        o_id=o_id,
        non_o_threshold=CUSTOM_NON_O_THRESHOLD,
        o_margin=CUSTOM_O_MARGIN,
    )

    strict_demo_entities = extract_strict_entities_from_merged_labels(merged_tokens, merged_labels)
    masked_demo_text = build_masked_text_from_strict_entities(custom_text, strict_demo_entities)

    print("\n========== Custom Sentence Prediction (Merged Words) ==========")
    print(f"Non-O threshold: {CUSTOM_NON_O_THRESHOLD:.2f} | O margin: {CUSTOM_O_MARGIN:.2f}")
    print(f"{'Index':<8}{'Word':<30}{'Predicted Label':<25}{'Confidence'}")
    print("-" * 95)
    for i, (tok, lab, conf) in enumerate(zip(merged_tokens, merged_labels, merged_confidences), start=1):
        print(f"{i:<8}{tok:<30}{lab:<25}{conf:.4f}")

    print("\n========== Strict Entity Output (all non-O merged word predictions) ==========")
    print(f"{'Index':<8}{'Entity Text':<35}{'Predicted Entity Type'}")
    print("-" * 80)
    for i, (entity_text, entity_label) in enumerate(strict_demo_entities, start=1):
        print(f"{i:<8}{entity_text:<35}{entity_label}")

    print("\n========== Masked Custom Text ==========")
    print(masked_demo_text)

    save_custom_entity_table_figure(
        strict_demo_entities=strict_demo_entities,
        output_path="custom_entity_output_table.png",
    )

    return {
        "custom_text": custom_text,
        "bert_tokens": bert_tokens,
        "bert_pred_labels": bert_pred_labels,
        "merged_tokens": merged_tokens,
        "merged_labels": merged_labels,
        "merged_confidences": merged_confidences,
        "strict_demo_entities": strict_demo_entities,
        "masked_demo_text": masked_demo_text,
    }