"""
Text Similarity Module

Implements cosine similarity for text comparison using TF-IDF
vectorization, without relying on external NLP libraries.
"""

import math
from collections import Counter
from typing import Dict, List, Optional, Tuple


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer with lowercasing and punctuation removal."""
    cleaned = ""
    for ch in text.lower():
        if ch.isalnum() or ch.isspace():
            cleaned += ch
        else:
            cleaned += " "
    return [token for token in cleaned.split() if len(token) > 0]


def compute_tf(tokens: List[str]) -> Dict[str, float]:
    """
    Compute Term Frequency (TF) for a list of tokens.

    Uses raw frequency normalized by document length.
    """
    counts = Counter(tokens)
    total = len(tokens)
    if total == 0:
        return {}
    return {term: count / total for term, count in counts.items()}


def compute_idf(documents: List[List[str]]) -> Dict[str, float]:
    """
    Compute Inverse Document Frequency (IDF) for a corpus.

    Uses log(N / df) where N is the number of documents
    and df is the number of documents containing the term.
    """
    n_docs = len(documents)
    if n_docs == 0:
        return {}

    df = Counter()
    for doc in documents:
        unique_terms = set(doc)
        for term in unique_terms:
            df[term] += 1

    idf = {}
    for term, doc_freq in df.items():
        idf[term] = math.log(n_docs / doc_freq) if doc_freq > 0 else 0.0

    return idf


def compute_tfidf(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    """
    Compute TF-IDF vector for a document.

    Args:
        tokens: Tokenized document.
        idf: Pre-computed IDF values.

    Returns:
        Dictionary mapping terms to TF-IDF weights.
    """
    tf = compute_tf(tokens)
    return {term: tf_val * idf.get(term, 0.0) for term, tf_val in tf.items()}


def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """
    Compute cosine similarity between two sparse vectors.

    Args:
        vec1: First vector as {term: weight}.
        vec2: Second vector as {term: weight}.

    Returns:
        Cosine similarity value between 0 and 1.
    """
    common_terms = set(vec1.keys()) & set(vec2.keys())

    dot_product = sum(vec1[t] * vec2[t] for t in common_terms)

    norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


class TFIDFSimilarity:
    """
    Computes text similarity using TF-IDF cosine similarity.

    Manages a corpus of documents and provides methods
    for computing pairwise similarity.
    """

    def __init__(self):
        self._documents: List[List[str]] = []
        self._raw_texts: List[str] = []
        self._idf: Dict[str, float] = {}
        self._tfidf_vectors: List[Dict[str, float]] = []

    def fit(self, texts: List[str]) -> "TFIDFSimilarity":
        """
        Fit the model on a corpus of texts.

        Args:
            texts: List of document strings.

        Returns:
            self
        """
        self._raw_texts = texts
        self._documents = [tokenize(text) for text in texts]
        self._idf = compute_idf(self._documents)
        self._tfidf_vectors = [compute_tfidf(doc, self._idf) for doc in self._documents]
        return self

    def similarity(self, idx1: int, idx2: int) -> float:
        """Compute similarity between two documents by index."""
        if not self._tfidf_vectors:
            raise RuntimeError("Model has not been fitted yet.")
        return cosine_similarity(self._tfidf_vectors[idx1], self._tfidf_vectors[idx2])

    def similarity_matrix(self) -> List[List[float]]:
        """Compute the full pairwise similarity matrix."""
        n = len(self._tfidf_vectors)
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1.0
                elif j > i:
                    sim = cosine_similarity(self._tfidf_vectors[i], self._tfidf_vectors[j])
                    matrix[i][j] = sim
                    matrix[j][i] = sim
        return matrix

    def query(self, text: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Find the most similar documents to a query text.

        Args:
            text: Query text.
            top_k: Number of top results to return.

        Returns:
            List of (document_index, similarity_score) tuples, sorted by similarity.
        """
        if not self._tfidf_vectors:
            raise RuntimeError("Model has not been fitted yet.")

        query_tokens = tokenize(text)
        query_vec = compute_tfidf(query_tokens, self._idf)

        scores = []
        for i, doc_vec in enumerate(self._tfidf_vectors):
            sim = cosine_similarity(query_vec, doc_vec)
            scores.append((i, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    @property
    def vocabulary_size(self) -> int:
        """Return the number of unique terms in the corpus."""
        return len(self._idf)

    @property
    def document_count(self) -> int:
        """Return the number of documents in the corpus."""
        return len(self._documents)
