import re
import unicodedata
import emoji
import pandas as pd
# from tqdm import tqdm
from typing import Callable, List, Union
from sklearn.base import BaseEstimator, TransformerMixin


class StringCleanerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, batch_functions: List[Callable[[List[str]], List[str]]]):
        self.batch_functions = batch_functions

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("Starting transformation in StringCleanerTransformer...")
        X = ensure_string_series(X).tolist()
        for function in self.batch_functions:
            X = function(X)
        print("Finished transformation in StringCleanerTransformer.")
        return X


class StringCleaner:
    @staticmethod
    def remove_line_breaks(texts: List[str]) -> List[str]:
        return [re.sub(r'[\n\r]', ' ', t) for t in texts]

    @staticmethod
    def remove_links(texts: List[str]) -> List[str]:
        pattern = r'http[s]?://(?:[a-zA-Z0-9$-_@.&+!*(),]|(?:%[0-9a-fA-F]{2}))+'
        return [re.sub(pattern, ' link ', t) for t in texts]

    @staticmethod
    def normalize_dates(texts: List[str]) -> List[str]:
        pattern = r'([0-2][0-9]|(3)[0-1])(\/|\.)(((0)[0-9])|((1)[0-2]))(\/|\.)\d{2,4}'
        return [re.sub(pattern, ' date ', t) for t in texts]

    @staticmethod
    def normalize_currency(texts: List[str]) -> List[str]:
        pattern = r'[R]{0,1}\$[ ]{0,}\d+(,|\.)\d+'
        return [re.sub(pattern, ' money ', t) for t in texts]

    @staticmethod
    def replace_numbers(texts: List[str]) -> List[str]:
        return [re.sub(r'[0-9]+', ' number ', t) for t in texts]

    @staticmethod
    def replace_negations(texts: List[str]) -> List[str]:
        pattern = r'([nN][ãÃaA][oO]|[ñÑ]| [nN])'
        return [re.sub(pattern, ' negation ', t) for t in texts]

    @staticmethod
    def remove_special_characters(texts: List[str]) -> List[str]:
        return [re.sub(r'\W', ' ', t) for t in texts]

    @staticmethod
    def normalize_whitespace(texts: List[str]) -> List[str]:
        step1 = [re.sub(r'\s+', ' ', t) for t in texts]
        return [re.sub(r'[ \t]+$', '', t) for t in step1]

    @staticmethod
    def replace_emojis(texts: List[str]) -> List[str]:
        return [emoji.replace_emoji(t, replace=' emoji ') for t in texts]

    @staticmethod
    def normalize_repetitions(texts: List[str]) -> List[str]:
        return [re.sub(r'(.)\1{2,}', r'\1\1', t) for t in texts]

    @staticmethod
    def remove_accents(texts: List[str]) -> List[str]:
        def strip_accents(text: str) -> str:
            return ''.join(
                c for c in unicodedata.normalize('NFD', text)
                if unicodedata.category(c) != 'Mn'
            )
        return [strip_accents(t) for t in texts]

    @staticmethod
    def replace_slang(texts: List[str]) -> List[str]:
        slang_dict = {
            'vc': 'you',
            'pq': 'because',
            'blz': 'ok',
            'n': 'no',
            'td': 'all',
            'q': 'that',
            'kd': 'where',
            'msg': 'message',
            'obg': 'thanks',
            'vlw': 'cheers',
            'aki': 'here',
        }

        def replace(text: str) -> str:
            for slang, standard in slang_dict.items():
                text = re.sub(rf'\b{slang}\b', standard, text, flags=re.IGNORECASE)
            return text

        return [replace(t) for t in texts]


class StopwordFilter(BaseEstimator, TransformerMixin):
    def __init__(self, stopwords: List[str]):
        self.stopwords = stopwords

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("Starting transformation in StopwordFilter...")
        X = ensure_string_series(X)
        result = [' '.join(self._remove_stopwords(text)) for text in X]
        print("Finished transformation in StopwordFilter.")
        return result

    def _remove_stopwords(self, text: str) -> List[str]:
        return [word.lower() for word in text.split() if word.lower() not in self.stopwords]



def ensure_string_series(X: Union[pd.Series, pd.DataFrame, List[str]]) -> pd.Series:
    if isinstance(X, pd.DataFrame):
        X = X.iloc[:, 0]
    if not isinstance(X, pd.Series):
        X = pd.Series(X)
    return X.astype(str)
