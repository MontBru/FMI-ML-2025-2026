import unittest
import numpy as np
from scipy.sparse import csr_matrix
from ml_lib.feature_extraction.text import TfidfVectorizer   # adjust import path if needed


class TestTfidfVectorizer(unittest.TestCase):

    def setUp(self):
        self.corpus = [
            "this is a test",
            "this is another test",
            "completely different words"
        ]
        self.vectorizer = TfidfVectorizer()

    def test_fit_sets_vocab_and_idf(self):
        self.vectorizer.fit(self.corpus)

        self.assertTrue(hasattr(self.vectorizer, "vocab"))
        self.assertTrue(hasattr(self.vectorizer, "idf"))
        self.assertIsInstance(self.vectorizer.vocab, dict)
        self.assertIsInstance(self.vectorizer.idf, np.ndarray)

    def test_vocab_size(self):
        self.vectorizer.fit(self.corpus)
        vocab = self.vectorizer.vocab

        expected_words = set("this is a test another completely different words".split())
        self.assertEqual(set(vocab.keys()), expected_words)

    def test_transform_shape(self):
        self.vectorizer.fit(self.corpus)
        X = self.vectorizer.transform(self.corpus)

        self.assertIsInstance(X, csr_matrix)
        self.assertEqual(X.shape, (3, len(self.vectorizer.vocab)))

    def test_fit_transform_equivalence(self):
        X1 = self.vectorizer.fit_transform(self.corpus)

        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(self.corpus)
        X2 = self.vectorizer.transform(self.corpus)

        self.assertTrue(np.allclose(X1.toarray(), X2.toarray()))

    def test_unknown_words_ignored(self):
        self.vectorizer.fit(self.corpus)

        new_corpus = ["this word does not exist"]
        X = self.vectorizer.transform(new_corpus)

        self.assertEqual(X.shape, (1, len(self.vectorizer.vocab)))
        self.assertGreater(X.nnz, 0)  # "this" exists
        self.assertEqual(
            X.nnz,
            1   # only "this" should be counted
        )

    def test_tfidf_non_negative(self):
        X = self.vectorizer.fit_transform(self.corpus)
        self.assertTrue((X.data >= 0).all())

    def test_empty_document(self):
        self.vectorizer.fit(self.corpus)

        X = self.vectorizer.transform([""])
        self.assertEqual(X.nnz, 0)
        self.assertEqual(X.shape, (1, len(self.vectorizer.vocab)))

    def test_idf_values(self):
        self.vectorizer.fit(self.corpus)
        idf = self.vectorizer.idf

        self.assertEqual(idf.shape[0], len(self.vectorizer.vocab))
        self.assertTrue(np.all(idf > 0))

    def test_sparse_output(self):
        X = self.vectorizer.fit_transform(self.corpus)
        self.assertIsInstance(X, csr_matrix)
        self.assertLess(X.nnz, X.shape[0] * X.shape[1])


if __name__ == "__main__":
    unittest.main()
