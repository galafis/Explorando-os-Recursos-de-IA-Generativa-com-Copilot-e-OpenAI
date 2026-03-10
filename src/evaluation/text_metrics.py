"""
Text Evaluation Metrics

Implements BLEU score and ROUGE score for evaluating
generated text quality against reference texts.
"""

from collections import Counter
from typing import Dict, List, Optional, Tuple
import math


def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization with lowercasing."""
    return text.lower().split()


def _get_ngrams(tokens: List[str], n: int) -> Counter:
    """Extract n-grams from a token list."""
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu_score(reference: str, hypothesis: str, max_n: int = 4,
               weights: Optional[List[float]] = None) -> Dict[str, float]:
    """
    Compute BLEU score between a reference and hypothesis text.

    Args:
        reference: Reference (ground truth) text.
        hypothesis: Generated (hypothesis) text.
        max_n: Maximum n-gram order (default 4).
        weights: Weights for each n-gram order (default uniform).

    Returns:
        Dictionary with 'bleu', 'precisions', 'brevity_penalty', and 'length_ratio'.
    """
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)

    if len(hyp_tokens) == 0:
        return {"bleu": 0.0, "precisions": [0.0] * max_n,
                "brevity_penalty": 0.0, "length_ratio": 0.0}

    if weights is None:
        weights = [1.0 / max_n] * max_n

    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = _get_ngrams(ref_tokens, n)
        hyp_ngrams = _get_ngrams(hyp_tokens, n)

        if len(hyp_ngrams) == 0:
            precisions.append(0.0)
            continue

        clipped_count = 0
        for ngram, count in hyp_ngrams.items():
            clipped_count += min(count, ref_ngrams.get(ngram, 0))

        precision = clipped_count / sum(hyp_ngrams.values())
        precisions.append(precision)

    length_ratio = len(hyp_tokens) / len(ref_tokens) if len(ref_tokens) > 0 else 0.0

    if length_ratio < 1.0 and length_ratio > 0:
        bp = math.exp(1 - 1.0 / length_ratio)
    else:
        bp = 1.0

    log_avg = 0.0
    for w, p in zip(weights, precisions):
        if p > 0:
            log_avg += w * math.log(p)
        else:
            return {"bleu": 0.0, "precisions": precisions,
                    "brevity_penalty": bp, "length_ratio": length_ratio}

    bleu = bp * math.exp(log_avg)

    return {
        "bleu": bleu,
        "precisions": precisions,
        "brevity_penalty": bp,
        "length_ratio": length_ratio,
    }


def rouge_n_score(reference: str, hypothesis: str, n: int = 1) -> Dict[str, float]:
    """
    Compute ROUGE-N score (recall, precision, F1) for n-grams.

    Args:
        reference: Reference text.
        hypothesis: Generated text.
        n: N-gram order (default 1 for ROUGE-1).

    Returns:
        Dictionary with 'precision', 'recall', and 'f1'.
    """
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)

    ref_ngrams = _get_ngrams(ref_tokens, n)
    hyp_ngrams = _get_ngrams(hyp_tokens, n)

    ref_total = sum(ref_ngrams.values())
    hyp_total = sum(hyp_ngrams.values())

    if ref_total == 0 or hyp_total == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    overlap = 0
    for ngram, count in ref_ngrams.items():
        overlap += min(count, hyp_ngrams.get(ngram, 0))

    precision = overlap / hyp_total
    recall = overlap / ref_total

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def rouge_l_score(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Compute ROUGE-L score based on longest common subsequence.

    Args:
        reference: Reference text.
        hypothesis: Generated text.

    Returns:
        Dictionary with 'precision', 'recall', and 'f1'.
    """
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)

    if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs_length = _lcs_length(ref_tokens, hyp_tokens)

    precision = lcs_length / len(hyp_tokens)
    recall = lcs_length / len(ref_tokens)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """Compute the length of the longest common subsequence."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def evaluate_text(reference: str, hypothesis: str) -> Dict[str, Dict[str, float]]:
    """
    Run all evaluation metrics on a reference-hypothesis pair.

    Args:
        reference: Reference text.
        hypothesis: Generated text.

    Returns:
        Dictionary with all metric results.
    """
    return {
        "bleu": bleu_score(reference, hypothesis),
        "rouge_1": rouge_n_score(reference, hypothesis, n=1),
        "rouge_2": rouge_n_score(reference, hypothesis, n=2),
        "rouge_l": rouge_l_score(reference, hypothesis),
    }
