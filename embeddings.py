"""
embeddings.py
-------------
Lightweight sentence representations built with TF-IDF + Truncated SVD
(a.k.a. LSA). This keeps the whole pipeline runnable fully offline (no
pretrained transformer weights need to be downloaded), while still giving
a semantic-ish space in which paraphrases of the same usage cluster
together -- which is all the downstream clustering / MMR-selection /
diversity-metric code needs.

Swap-in note: in a networked environment, replace `SentenceEmbedder`
with a `sentence-transformers` encoder (e.g. all-MiniLM-L6-v2) without
touching any other module -- every downstream component only depends on
the `.transform(list[str]) -> np.ndarray` interface.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


class SentenceEmbedder:
    def __init__(self, n_components: int = 32, random_state: int = 42):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=1, stop_words="english", sublinear_tf=True
        )
        self.n_components = n_components
        self.random_state = random_state
        self.svd = None

    def fit(self, sentences):
        X = self.vectorizer.fit_transform(sentences)
        k = min(self.n_components, max(2, min(X.shape) - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=self.random_state)
        self.svd.fit(X)
        return self

    def transform(self, sentences) -> np.ndarray:
        X = self.vectorizer.transform(sentences)
        Z = self.svd.transform(X)
        return normalize(Z)

    def fit_transform(self, sentences) -> np.ndarray:
        self.fit(sentences)
        return self.transform(sentences)
