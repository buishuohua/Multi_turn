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
    Truncate sentences in data either equally or proportionally.
    
    Args:
        data: DataFrame containing sentences
        tokenizer: HuggingFace tokenizer
        max_tokens: Maximum number of tokens allowed
        which: Column name containing sentences
        method: "equal" or "ratio" for truncation method
    
    Returns:
        List of truncated sentence lists
    """
    sentences = data[which]
    truncated_data = []

    if method == "equal":
        for sent_series in sentences:
            # Calculate tokens per sentence
            num_sentences = len(sent_series)
            num_seps = num_sentences - 1
            tokens_per_sent = (max_tokens - num_seps - 2) // num_sentences  # -2 for [CLS] and final [SEP]

            # Truncate each sentence
            truncated_sents = []
            for sent in sent_series:
                tokens = tokenizer(
                    sent,
                    truncation=True,
                    max_length=tokens_per_sent,
                    add_special_tokens=False
                )
                decoded = tokenizer.decode(tokens['input_ids'])
                truncated_sents.append(decoded)

            truncated_data.append(truncated_sents)

    elif method == "ratio":
        for sent_series in sentences:
            # Calculate original token lengths
            token_lengths = [len(tokenizer.encode(sent)) for sent in sent_series]
            total_original_tokens = sum(token_lengths)

            # Account for special tokens
            num_seps = len(sent_series) - 1
            available_tokens = max_tokens - num_seps - 2  # -2 for [CLS] and final [SEP]

            # Calculate proportional lengths
            proportions = [length / total_original_tokens for length in token_lengths]
            allocated_tokens = [max(1, int(prop * available_tokens)) for prop in proportions]

            # Truncate each sentence
            truncated_sents = []
            for sent, max_len in zip(sent_series, allocated_tokens):
                tokens = tokenizer(
                    sent,
                    truncation=True,
                    max_length=max_len,
                    add_special_tokens=False
                )
                decoded = tokenizer.decode(tokens['input_ids'])
                truncated_sents.append(decoded)

            truncated_data.append(truncated_sents)

    else:
        raise ValueError("Truncate method must be equal or ratio")

    data[f"{which}_truncated_lst"] = truncated_data
    data[f"{which}_truncated"] = data[f"{which}_truncated_lst"].apply(lambda x: " ".join(x))
    data.drop(columns=[f"{which}_truncated_lst"], inplace=True)
    data.reset_index(drop=True, inplace=True)

    return data


if __name__ == '__main__':
    import pandas as pd
    from load_split import loader
    from models.Embedding.BERT import BERTs

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
