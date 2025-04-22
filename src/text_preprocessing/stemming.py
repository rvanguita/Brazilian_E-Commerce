from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from typing import Any, List, Union


class StemmerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, stemmer: Any):
        self.stemmer = stemmer

    def fit(self, X: Union[pd.Series, pd.DataFrame, List[str]], y=None):
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame, List[str]], y=None) -> List[str]:
        print("Starting transformation in StemmerTransformer...")
        X = ensure_string_series(X)
        result = [' '.join(self._apply_stemming(text)) for text in X]
        print("Finished transformation in StemmerTransformer.")
        return result

    def _apply_stemming(self, text: str) -> List[str]:
        return [self.stemmer.stem(word) for word in text.split()]



def ensure_string_series(X: Union[pd.Series, pd.DataFrame, List[str]]) -> pd.Series:
    if isinstance(X, pd.DataFrame):
        X = X.iloc[:, 0]
    if not isinstance(X, pd.Series):
        X = pd.Series(X)
    return X.astype(str)
