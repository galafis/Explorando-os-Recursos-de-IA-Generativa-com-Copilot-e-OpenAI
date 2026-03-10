"""
Simple Tokenizer Utilities

Provides whitespace-based tokenization with token counting,
vocabulary building, and basic text statistics.
"""

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple


class SimpleTokenizer:
    """
    Whitespace tokenizer with configurable preprocessing.

    Supports lowercasing, punctuation handling, stopword removal,
    and vocabulary management.
    """

    DEFAULT_STOPWORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "can", "could", "must", "and", "but", "or",
        "nor", "not", "so", "yet", "for", "at", "by", "from", "in", "into",
        "of", "on", "to", "with", "as", "if", "it", "its", "this", "that",
        "these", "those", "i", "you", "he", "she", "we", "they", "me", "him",
        "her", "us", "them", "my", "your", "his", "our", "their",
    }

    def __init__(self, lowercase: bool = True, remove_punctuation: bool = True,
                 remove_stopwords: bool = False, min_token_length: int = 1):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.min_token_length = min_token_length
        self._vocabulary: Dict[str, int] = {}
        self._total_tokens = 0

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a text string.

        Args:
            text: Input text to tokenize.

        Returns:
            List of tokens.
        """
        if self.lowercase:
            text = text.lower()

        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)

        tokens = text.split()

        if self.min_token_length > 1:
            tokens = [t for t in tokens if len(t) >= self.min_token_length]

        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.DEFAULT_STOPWORDS]

        return tokens

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in the text."""
        return len(self.tokenize(text))

    def token_frequencies(self, text: str) -> Dict[str, int]:
        """Return token frequency counts for the text."""
        return dict(Counter(self.tokenize(text)))

    def build_vocabulary(self, texts: List[str], max_vocab_size: Optional[int] = None) -> Dict[str, int]:
        """
        Build a vocabulary from a list of texts.

        Args:
            texts: List of text strings.
            max_vocab_size: Maximum vocabulary size (most frequent tokens kept).

        Returns:
            Dictionary mapping tokens to their frequency.
        """
        all_tokens = []
        for text in texts:
            all_tokens.extend(self.tokenize(text))

        self._total_tokens = len(all_tokens)
        counts = Counter(all_tokens)

        if max_vocab_size is not None:
            most_common = counts.most_common(max_vocab_size)
            self._vocabulary = dict(most_common)
        else:
            self._vocabulary = dict(counts)

        return self._vocabulary

    def text_statistics(self, text: str) -> Dict[str, object]:
        """
        Compute text statistics.

        Args:
            text: Input text.

        Returns:
            Dictionary with token_count, unique_tokens, avg_token_length,
            char_count, and most_common tokens.
        """
        tokens = self.tokenize(text)
        unique = set(tokens)

        if len(tokens) == 0:
            return {
                "token_count": 0,
                "unique_tokens": 0,
                "avg_token_length": 0.0,
                "char_count": len(text),
                "type_token_ratio": 0.0,
                "most_common": [],
            }

        avg_length = sum(len(t) for t in tokens) / len(tokens)
        most_common = Counter(tokens).most_common(10)

        return {
            "token_count": len(tokens),
            "unique_tokens": len(unique),
            "avg_token_length": round(avg_length, 2),
            "char_count": len(text),
            "type_token_ratio": round(len(unique) / len(tokens), 4),
            "most_common": most_common,
        }

    def batch_tokenize(self, texts: List[str]) -> List[List[str]]:
        """Tokenize multiple texts at once."""
        return [self.tokenize(text) for text in texts]

    def batch_count(self, texts: List[str]) -> List[int]:
        """Count tokens in multiple texts at once."""
        return [self.count_tokens(text) for text in texts]

    @property
    def vocabulary_size(self) -> int:
        """Return the current vocabulary size."""
        return len(self._vocabulary)

    @property
    def total_tokens(self) -> int:
        """Return the total token count from vocabulary building."""
        return self._total_tokens

    def encode(self, text: str) -> List[int]:
        """
        Encode text as a list of vocabulary indices.

        Tokens not in vocabulary are mapped to 0 (unknown).
        """
        if not self._vocabulary:
            raise RuntimeError("Vocabulary has not been built yet. Call build_vocabulary first.")

        vocab_to_idx = {token: idx + 1 for idx, token in enumerate(self._vocabulary)}
        tokens = self.tokenize(text)
        return [vocab_to_idx.get(token, 0) for token in tokens]

    def decode(self, indices: List[int]) -> List[str]:
        """
        Decode vocabulary indices back to tokens.

        Index 0 maps to '<UNK>'.
        """
        if not self._vocabulary:
            raise RuntimeError("Vocabulary has not been built yet.")

        idx_to_vocab = {idx + 1: token for idx, token in enumerate(self._vocabulary)}
        idx_to_vocab[0] = "<UNK>"
        return [idx_to_vocab.get(idx, "<UNK>") for idx in indices]
