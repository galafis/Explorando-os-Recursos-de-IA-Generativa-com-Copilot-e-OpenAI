"""
Tests for evaluation metrics, similarity, and tokenizer modules.
"""

import pytest
from src.evaluation.text_metrics import bleu_score, rouge_n_score, rouge_l_score, evaluate_text
from src.embeddings.similarity import (
    tokenize, compute_tf, compute_idf, cosine_similarity, TFIDFSimilarity
)
from src.utils.tokenizer import SimpleTokenizer


class TestBLEU:
    def test_exact_match(self):
        result = bleu_score("the cat sat on the mat", "the cat sat on the mat")
        assert result["bleu"] == pytest.approx(1.0, abs=0.01)

    def test_no_match(self):
        result = bleu_score("the cat sat on the mat", "dogs run in the park quickly")
        assert result["bleu"] < 0.2

    def test_partial_match(self):
        result = bleu_score("the cat sat on the mat", "the cat sat")
        assert 0.0 < result["bleu"] < 1.0

    def test_empty_hypothesis(self):
        result = bleu_score("the cat sat", "")
        assert result["bleu"] == 0.0

    def test_brevity_penalty(self):
        result = bleu_score("the cat sat on the mat", "cat")
        assert result["brevity_penalty"] < 1.0


class TestROUGE:
    def test_rouge1_exact(self):
        result = rouge_n_score("the cat sat on the mat", "the cat sat on the mat", n=1)
        assert result["f1"] == pytest.approx(1.0, abs=0.01)

    def test_rouge1_partial(self):
        result = rouge_n_score("the cat sat on the mat", "the dog sat", n=1)
        assert 0.0 < result["f1"] < 1.0

    def test_rouge2(self):
        result = rouge_n_score("the cat sat on the mat", "the cat sat", n=2)
        assert result["recall"] > 0

    def test_rouge_l_exact(self):
        result = rouge_l_score("the cat sat", "the cat sat")
        assert result["f1"] == pytest.approx(1.0, abs=0.01)

    def test_rouge_l_partial(self):
        result = rouge_l_score("the cat sat on the mat", "the dog sat on a mat")
        assert 0.0 < result["f1"] < 1.0

    def test_rouge_empty(self):
        result = rouge_n_score("hello world", "", n=1)
        assert result["f1"] == 0.0


class TestEvaluateText:
    def test_returns_all_metrics(self):
        result = evaluate_text("hello world", "hello world")
        assert "bleu" in result
        assert "rouge_1" in result
        assert "rouge_2" in result
        assert "rouge_l" in result


class TestTokenize:
    def test_basic(self):
        tokens = tokenize("Hello World!")
        assert tokens == ["hello", "world"]

    def test_punctuation_removal(self):
        tokens = tokenize("It's a test, right?")
        assert "it" in tokens
        assert "test" in tokens


class TestTFIDF:
    def test_tf(self):
        tokens = ["a", "b", "a", "c"]
        tf = compute_tf(tokens)
        assert tf["a"] == pytest.approx(0.5)

    def test_idf(self):
        docs = [["a", "b"], ["b", "c"], ["a", "c"]]
        idf = compute_idf(docs)
        assert idf["b"] > 0

    def test_cosine_identical(self):
        vec = {"a": 1.0, "b": 2.0}
        assert cosine_similarity(vec, vec) == pytest.approx(1.0, abs=0.001)

    def test_cosine_orthogonal(self):
        vec1 = {"a": 1.0}
        vec2 = {"b": 1.0}
        assert cosine_similarity(vec1, vec2) == pytest.approx(0.0)


class TestTFIDFSimilarity:
    def test_fit_and_query(self):
        model = TFIDFSimilarity()
        model.fit(["machine learning", "deep learning neural", "cooking recipes food"])
        results = model.query("learning algorithms", top_k=2)
        assert len(results) == 2
        assert results[0][1] >= results[1][1]

    def test_similarity_matrix(self):
        model = TFIDFSimilarity()
        model.fit(["cat dog", "dog cat", "fish bird"])
        matrix = model.similarity_matrix()
        assert matrix[0][0] == pytest.approx(1.0)
        assert matrix[0][1] > matrix[0][2]

    def test_properties(self):
        model = TFIDFSimilarity()
        model.fit(["hello world", "foo bar"])
        assert model.document_count == 2
        assert model.vocabulary_size > 0


class TestSimpleTokenizer:
    def test_basic_tokenization(self):
        tok = SimpleTokenizer()
        tokens = tok.tokenize("Hello, World!")
        assert "hello" in tokens
        assert "world" in tokens

    def test_count_tokens(self):
        tok = SimpleTokenizer()
        assert tok.count_tokens("one two three") == 3

    def test_stopword_removal(self):
        tok = SimpleTokenizer(remove_stopwords=True)
        tokens = tok.tokenize("the cat is on the mat")
        assert "the" not in tokens
        assert "cat" in tokens

    def test_min_token_length(self):
        tok = SimpleTokenizer(min_token_length=3)
        tokens = tok.tokenize("I am a developer")
        assert "i" not in tokens
        assert "am" not in tokens

    def test_build_vocabulary(self):
        tok = SimpleTokenizer()
        vocab = tok.build_vocabulary(["hello world", "hello python"])
        assert "hello" in vocab
        assert vocab["hello"] == 2

    def test_text_statistics(self):
        tok = SimpleTokenizer()
        stats = tok.text_statistics("hello world hello")
        assert stats["token_count"] == 3
        assert stats["unique_tokens"] == 2
