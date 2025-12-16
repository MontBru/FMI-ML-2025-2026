from scipy.sparse import csr_matrix
from collections import Counter
import numpy as np


class TfidfVectorizer:
    def __init__(self):
        pass


    def get_term_count_and_vocab(self, corpus):
        # corpus: list of strings
        n_docs = len(corpus)

        # 1. Tokenize and count words per document
        tokenized = [doc.split() for doc in corpus]
        doc_counters = [Counter(doc) for doc in tokenized]

        # 2. Build vocabulary
        vocab = {}
        for counter in doc_counters:
            for word in counter:
                if word not in vocab:
                    vocab[word] = len(vocab)

        n_words = len(vocab)

        # 3. Build sparse matrix data
        rows = []
        cols = []
        data = []

        for doc_id, counter in enumerate(doc_counters):
            for word, count in counter.items():
                rows.append(doc_id)
                cols.append(vocab[word])
                data.append(count)

        # 4. Create CSR matrix
        term_frequency = csr_matrix(
            (data, (rows, cols)),
            shape=(n_docs, n_words),
            dtype=int
        )

        return term_frequency, vocab

    def get_term_frequency(self, term_count):
        row_sums = term_count.sum(axis=1)      # (N, 1) sparse
        row_sums[row_sums == 0] = 1
        return term_count.multiply(1 / row_sums)

    
    def get_idf(self, term_count):
        N_docs = term_count.shape[0]
        df = np.asarray((term_count > 0).sum(axis=0)).ravel()
        idf = np.log((N_docs + 1) / (df + 1)) + 1
        return idf
    
    def get_term_count(self, corpus, vocab):
        rows = []
        cols = []
        data = []

        for doc_id, doc in enumerate(corpus):
            counter = Counter(doc.split())
            for word, count in counter.items():
                if word in vocab:           # ignore unknown words
                    rows.append(doc_id)
                    cols.append(vocab[word])
                    data.append(count)

        return csr_matrix(
            (data, (rows, cols)),
            shape=(len(corpus), len(vocab)),
            dtype=int
        )


    def fit(self, corpus):
        #corpus is a list of strings
        term_count, vocab = self.get_term_count_and_vocab(corpus)
        
        inverse_document_frequency = self.get_idf(term_count)
        
        self.idf = inverse_document_frequency
        self.vocab = vocab

    def transform(self, corpus):
        term_count = self.get_term_count(corpus, self.vocab)
        tf = self.get_term_frequency(term_count)
        return tf.multiply(self.idf).tocsr()


    def fit_transform(self, corpus):
        self.fit(corpus)
        return self.transform(corpus)


