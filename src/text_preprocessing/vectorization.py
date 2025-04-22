from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from typing import Any, Union


class TextVectorizerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, vectorizer: Any):
        self.vectorizer = vectorizer

    def fit(self, X: Union[pd.Series, pd.DataFrame, list], y=None):
        X = ensure_string_series(X)
        self.vectorizer.fit(X)
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame, list], y=None):
        X = ensure_string_series(X)
        return self.vectorizer.transform(X)  # returns sparse matrix


def ensure_string_series(X: Union[pd.Series, pd.DataFrame, list]) -> pd.Series:
    if isinstance(X, pd.DataFrame):
        X = X.iloc[:, 0]
    if not isinstance(X, pd.Series):
        X = pd.Series(X)
    return X.astype(str)
