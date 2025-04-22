import pandas as pd


from sklearn.feature_extraction.text import CountVectorizer


class FeatureExtractionUtils:
    """
    A utility class that provides static methods for feature extraction from text data.

    This class is designed to assist in transforming raw text into structured features 
    that can be used in machine learning models. It supports document-term matrix creation 
    and n-gram frequency analysis, which are foundational techniques in natural language processing (NLP).

    The methods are stateless and can be used independently of any pipeline, making them suitable 
    for both exploratory analysis and production-level preprocessing.

    Methods:
    --------
    extract_features_from_corpus(corpus: list[str], vectorizer, return_df: bool = False)
        Transforms a list of text documents into a document-term matrix using a given 
        scikit-learn vectorizer (e.g., TfidfVectorizer or CountVectorizer).
        Optionally returns a pandas DataFrame with feature names as columns.

    get_top_ngrams(texts: list[str], ngram_range: tuple, top_n: int = -1, stopwords_list: list[str] = None) -> pd.DataFrame
        Extracts and ranks the most frequent n-grams in a list of text documents.
        Supports custom n-gram ranges and optional stopword filtering.

    Examples:
    ---------
    # Example 1: Extracting features
    X, df_features = FeatureExtractionUtils.extract_features_from_corpus(
        corpus=documents,
        vectorizer=TfidfVectorizer(),
        return_df=True
    )

    # Example 2: Top bigrams
    top_bigrams = FeatureExtractionUtils.get_top_ngrams(
        texts=documents,
        ngram_range=(2, 2),
        top_n=10,
        stopwords_list=stopwords.words('portuguese')
    )
    """

    @staticmethod
    def extract_features_from_corpus(corpus: list[str], vectorizer, return_df: bool = False):
        """
        Converts a text corpus into a document-term matrix.

        Parameters
        ----------
        corpus : list of str
            List of documents.
        vectorizer : sklearn vectorizer
            Object with `fit_transform()` and `get_feature_names_out()` methods.
        return_df : bool
            If True, returns a DataFrame with feature names as columns.

        Returns
        -------
        tuple
            - ndarray: Document-term matrix.
            - DataFrame or None: DataFrame with features (if return_df=True).
        """
        features_matrix = vectorizer.fit_transform(corpus).toarray()
        feature_names = vectorizer.get_feature_names_out()
        features_df = pd.DataFrame(features_matrix, columns=feature_names) if return_df else None
        return features_matrix, features_df

    @staticmethod
    def get_top_ngrams(texts: list[str], ngram_range: tuple, top_n: int = -1, stopwords_list: list[str] = None) -> pd.DataFrame:
        """
        Extracts the most frequent n-grams from a list of texts.

        Parameters
        ----------
        texts : list of str
            Corpus to analyze.
        ngram_range : tuple
            N-gram size, e.g., (1,1) for unigrams.
        top_n : int
            Max number of n-grams to return. -1 means all.
        stopwords_list : list of str
            Optional stopwords to ignore.

        Returns
        -------
        pd.DataFrame
            DataFrame with ['ngram', 'count'], sorted by frequency.
        """
        vectorizer = CountVectorizer(stop_words=stopwords_list, ngram_range=ngram_range).fit(texts)
        bow_matrix = vectorizer.transform(texts)
        ngram_counts = bow_matrix.sum(axis=0)

        # Extract and sort n-gram frequencies
        ngram_frequencies = [(ngram, ngram_counts[0, idx]) for ngram, idx in vectorizer.vocabulary_.items()]
        ngram_frequencies = sorted(ngram_frequencies, key=lambda x: x[1], reverse=True)

        # Slice top_n if needed
        top_ngrams = ngram_frequencies[:top_n] if top_n > 0 else ngram_frequencies

        return pd.DataFrame(top_ngrams, columns=['ngram', 'count'])


