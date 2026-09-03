from __future__ import annotations
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline


class CyberTextClassifier:
    """Bounded-memory baseline suitable for large vulnerability corpora."""

    def __init__(self, n_features: int = 2**18, random_state: int = 7):
        self.pipeline = Pipeline([
            ("features", HashingVectorizer(n_features=n_features, ngram_range=(1, 2), alternate_sign=False)),
            ("classifier", SGDClassifier(loss="log_loss", class_weight="balanced", random_state=random_state)),
        ])

    def fit(self, texts: list[str], labels: list[str]) -> "CyberTextClassifier":
        if len(texts) != len(labels) or len(set(labels)) < 2:
            raise ValueError("aligned examples from at least two classes are required")
        self.pipeline.fit(texts, labels)
        return self

    def predict(self, texts: list[str]):
        return self.pipeline.predict(texts)

    def predict_proba(self, texts: list[str]):
        return self.pipeline.predict_proba(texts)

