# -*- coding: utf-8 -*- 

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 2024/12/14 14:50
    @ Description:
"""


def token_stats_info(data, tokenizer, which):
    sentences = data[which]

    data["token_lengths"] = None
    data["total_original_tokens"] = None
    for i, sent_series in enumerate(sentences):
        token_lengths = [len(tokenizer.encode(sent)) for sent in sent_series]
        total_original_tokens = sum(token_lengths)
        data.at[i, "token_lengths"] = token_lengths
        data.at[i, "total_original_tokens"] = total_original_tokens

    data[f"total_original_tokens_{which}"] = pd.to_numeric(data["total_original_tokens"], errors='coerce')
    return data


def truncate(data, tokenizer, max_tokens, which, method):
    """
    Truncate sentences and store token IDs.
    """
    sentences = data[which].values
    tokenized_data = []
    truncated_texts = []

    for sent_series in sentences:
        num_sentences = len(sent_series)
        num_seps = num_sentences - 1

        if method == "equal":
            tokens_per_sent = (max_tokens - num_seps - 2) // num_sentences
            tokens_per_sent = min(tokens_per_sent, max_tokens // num_sentences)

            sent_tokens = []
            trunc_texts = []
            for sent in sent_series:
                tokens = tokenizer(
                    sent,
                    truncation=True,
                    padding="max_length",
                    max_length=tokens_per_sent,
                    add_special_tokens=False,
                    return_tensors='pt'
                )
                # Get the token IDs as a list
                tokens_list = tokens['input_ids'].squeeze(
                    0).tolist()  # Add squeeze(0)
                sent_tokens.append(tokens_list)
                trunc_texts.append(tokenizer.decode(tokens_list))

        elif method == "ratio":
            token_lengths = [len(tokenizer.encode(sent))
                             for sent in sent_series]
            total_original_tokens = sum(token_lengths)
            available_tokens = max_tokens - num_seps - 2

            proportions = [
                length/total_original_tokens for length in token_lengths]
            allocated_tokens = [min(max(1, int(prop * available_tokens)),
                                    max_tokens // num_sentences)
                                for prop in proportions]

            sent_tokens = []
            trunc_texts = []
            for sent, max_len in zip(sent_series, allocated_tokens):
                tokens = tokenizer(
                    sent,
                    truncation=True,
                    padding="max_length",
                    max_length=max_len,
                    add_special_tokens=False,
                    return_tensors='pt'
                )
                tokens_list = tokens['input_ids'].squeeze(
                    0).tolist()  # Add squeeze(0)
                sent_tokens.append(tokens_list)
                trunc_texts.append(tokenizer.decode(tokens_list))

        # Combine all tokens with [CLS] and [SEP]
        final_tokens = [tokenizer.cls_token_id]
        # Change variable name for clarity
        for idx, token_list in enumerate(sent_tokens):
            final_tokens.extend(token_list)  # Now extending with a list
            if idx < len(sent_tokens) - 1:
                final_tokens.append(tokenizer.sep_token_id)
        final_tokens.append(tokenizer.sep_token_id)

        # Truncate if too long
        if len(final_tokens) > max_tokens:
            final_tokens = final_tokens[:max_tokens-1]
            final_tokens.append(tokenizer.sep_token_id)

        # Pad if necessary
        pad_length = max_tokens - len(final_tokens)
        if pad_length > 0:
            final_tokens.extend([tokenizer.pad_token_id] * pad_length)

        tokenized_data.append(final_tokens)
        truncated_texts.append(" [SEP] ".join(trunc_texts))

    # Store both tokenized IDs and truncated text
    data = data.copy()
    data[f"{which}_tokenized"] = tokenized_data
    data[f"{which}_truncated"] = truncated_texts

    return data


if __name__ == '__main__':
    import pandas as pd
    from load_split import loader
    from models.Embedding.BERT_tokenizer import BERTs

    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)

    data = loader()
    # data = token_stats_info(data, Tokenizer.BERT_base_uncased, "question")
    #
    # print(data["total_original_tokens_question"].describe().round(0))
    # print(data["total_original_tokens_question"].quantile([0.8, 0.85, 0.9, 0.95]).round(0))

    data = truncate(data, BERTs.BERT_base_uncased.tokenizer, 256, "question", "equal")
    print(data["question_truncated"].head())
