import pandas as pd
from typing import List, Tuple, Optional, Union
from sklearn.feature_extraction.text import CountVectorizer


class FeatureExtractionUtils:
    """
    Utility class for text feature extraction.

    Provides static methods for:
    - Document-term matrix creation
    - Top n-gram frequency analysis
    """

    @staticmethod
    def extract_features_from_corpus(
        corpus: List[str],
        vectorizer,
        return_df: bool = False
    ) -> Tuple:
        """
        Generates a document-term matrix using the provided vectorizer.

        Parameters:
        - corpus: List of documents.
        - vectorizer: Must implement `fit_transform` and `get_feature_names_out`.
        - return_df: Whether to return the matrix as a DataFrame.

        Returns:
        - features_matrix: Numpy array of shape (n_samples, n_features).
        - features_df: Optional DataFrame if return_df is True.
        """
        features_matrix = vectorizer.fit_transform(corpus).toarray()
        feature_names = vectorizer.get_feature_names_out()
        features_df = pd.DataFrame(features_matrix, columns=feature_names) if return_df else None
        return features_matrix, features_df

    @staticmethod
    def get_top_ngrams(
        texts: List[str],
        ngram_range: Tuple[int, int],
        top_n: int = -1,
        stopwords_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Extracts and ranks the most frequent n-grams from the corpus.

        Parameters:
        - texts: List of input strings.
        - ngram_range: Tuple (min_n, max_n) for n-gram sizes.
        - top_n: Number of top n-grams to return; -1 for all.
        - stopwords_list: Optional list of stopwords.

        Returns:
        - DataFrame with columns ['ngram', 'count'] sorted by frequency.
        """
        vectorizer = CountVectorizer(stop_words=stopwords_list, ngram_range=ngram_range)
        bow_matrix = vectorizer.fit_transform(texts)
        ngram_counts = bow_matrix.sum(axis=0).A1  # Convert to 1D array
        ngrams = vectorizer.get_feature_names_out()

        ngram_freq = sorted(zip(ngrams, ngram_counts), key=lambda x: x[1], reverse=True)
        if top_n > 0:
            ngram_freq = ngram_freq[:top_n]

        return pd.DataFrame(ngram_freq, columns=['ngram', 'count'])
