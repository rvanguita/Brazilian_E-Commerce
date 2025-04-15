import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re

from nltk.stem import RSLPStemmer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin


import unicodedata
import emoji


class RegexPreprocessor(BaseEstimator, TransformerMixin):
    """
    Applies a series of regular expression-based transformations to a text corpus.
    
    Parameters:
    ----------
    regex_functions : dict
        A dictionary where keys are function names (str) and values are callable 
        functions that apply regex transformations to the input text.
    """
    
    def __init__(self, regex_functions):
        self.regex_functions = regex_functions  # Store the dictionary of regex functions
        
    def fit(self, X, y=None):
        # No fitting necessary, return self for pipeline compatibility
        return self
    
    def transform(self, X, y=None):
        # Apply each regex function sequentially to the input text
        for name, function in self.regex_functions.items():
            X = function(X)
        return X



class StopwordRemover(BaseEstimator, TransformerMixin):
    """
    Removes stopwords from each document in the text corpus.
    
    Parameters:
    ----------
    stopword_list : list
        A list of stopwords to be removed from the text.
    """
    
    def __init__(self, stopwords):
        self.stopwords = stopwords  # Save the list of stopwords
        
    def fit(self, data):
        # No fitting necessary, return self for pipeline compatibility
        return self
    
    def transform(self, data):
        # For each text in the corpus, remove stopwords using the external helper function
        return [' '.join(self.remove_stopwords(text)) for text in data]
    
    
    def remove_stopwords(self, text: str) -> list[str]:
        """
        Removes stopwords and converts words to lowercase.

        Parameters
        ----------
        text : str
            Input text.
        stopwords_list : list of str
            Words to exclude from the output.

        Returns
        -------
        list of str
            Tokens without stopwords, all in lowercase.
        """
        return [word.lower() for word in text.split() if word.lower() not in self.stopwords]



class StemmerTransformer(BaseEstimator, TransformerMixin):
    """
    Applies stemming to each word in the corpus using the provided stemmer.
    
    Parameters:
    ----------
    stemmer : object
        A stemmer object (e.g., from nltk or SnowballStemmer) with a `stem` method.
    """
    
    def __init__(self, stemmer):
        self.stemmer = stemmer  # Store the provided stemmer instance
    
    def fit(self, data):
        # No fitting required
        return self
    
    def transform(self, data):
        # Apply stemming to each document in the corpus using external helper function
        return [' '.join(self.apply_stemming(text=text)) for text in data]
    
    def apply_stemming(self, text: str) -> list[str]:
        """
        Applies stemming to each word in a text string.

        Parameters
        ----------
        text : str
            Input text string.
        stemmer : nltk.stem.api.StemmerI
            Any NLTK-compatible stemmer object.

        Returns
        -------
        list of str
            List of stemmed tokens.
        """
        return [self.stemmer.stem(word) for word in text.split()]



class TextVectorizer(BaseEstimator, TransformerMixin):
    """
    Converts a corpus of text into a matrix of numeric features using a vectorizer.
    
    Parameters:
    ----------
    vectorizer : object
        A vectorizer instance (e.g., TfidfVectorizer or CountVectorizer) from scikit-learn.
    """
    
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer  # Store the provided vectorizer object
        
    def fit(self, X, y=None):
        # Fit the vectorizer to the data
        return self
    
    def transform(self, X, y=None):
        # Transform the corpus into a numerical array (dense)
        return self.vectorizer.fit_transform(X).toarray()



class RegexInspector:
    """
    Utility class for inspecting regex patterns in text data.
    """

    @staticmethod
    def find_pattern_spans(pattern: str, texts: list) -> dict:
        """
        Finds all match spans for a regex pattern across a list of text strings.

        Parameters
        ----------
        pattern : str
            A regular expression pattern to search for.
        texts : list of str
            List of input texts.

        Returns
        -------
        dict
            A dictionary where keys are text indices (as strings) and values are lists of
            tuple spans (start, end) indicating where matches were found in each text.
        """
        compiled_pattern = re.compile(pattern)  # Compile regex for performance
        match_spans_by_text = {}

        for index, text in enumerate(texts):
            spans = [match.span() for match in compiled_pattern.finditer(text)]
            if spans:
                match_spans_by_text[f'Text idx {index}'] = spans  # Only keep if matches were found

        return match_spans_by_text
    
    def print_transformation_comparison(before_texts: list, after_texts: list, indexes: list):
        """
        Prints side-by-side comparison of texts before and after a transformation step.

        Parameters
        ----------
        before_texts : list or pandas.Series
            Original text data before transformation.
        after_texts : list or pandas.Series
            Transformed text data after applying a processing step.
        indexes : list of int
            List of indices of the texts to be displayed for comparison.
        """
        for i, idx in enumerate(indexes, 1):
            print(f'--- Text {i} (Index {idx}) ---\n')
            print(f'Before: \n{before_texts.iloc[idx]}\n')
            print(f'After: \n{after_texts.iloc[idx]}\n')
            print('-' * 50)


class RegexCleaner:
    """
    Collection of static methods for regex-based text cleaning.
    Each method receives a list of texts and returns the cleaned list.
    """

    @staticmethod
    def clean_breaklines(texts: list) -> list:
        return [re.sub(r'[\n\r]', ' ', t) for t in texts]

    @staticmethod
    def clean_links(texts: list) -> list:
        pattern = r'http[s]?://(?:[a-zA-Z0-9$-_@.&+!*(),]|(?:%[0-9a-fA-F]{2}))+'
        return [re.sub(pattern, ' link ', t) for t in texts]

    @staticmethod
    def clean_dates(texts: list) -> list:
        pattern = r'([0-2][0-9]|(3)[0-1])(\/|\.)(((0)[0-9])|((1)[0-2]))(\/|\.)\d{2,4}'
        return [re.sub(pattern, ' data ', t) for t in texts]

    @staticmethod
    def clean_currency(texts: list) -> list:
        pattern = r'[R]{0,1}\$[ ]{0,}\d+(,|\.)\d+'
        return [re.sub(pattern, ' dinheiro ', t) for t in texts]

    @staticmethod
    def clean_numbers(texts: list) -> list:
        return [re.sub('[0-9]+', ' numero ', r) for r in texts]

    @staticmethod
    def clean_negations(texts: list) -> list:
        pattern = r'([nN][ãÃaA][oO]|[ñÑ]| [nN])'
        return [re.sub(pattern, ' negação ', t) for t in texts]

    @staticmethod
    def clean_special_characters(texts: list) -> list:
        return [re.sub('\W', ' ', r) for r in texts]

    @staticmethod
    def clean_whitespace(texts: list) -> list:
        white_spaces = [re.sub('\s+', ' ', r) for r in texts]
        white_spaces_end = [re.sub('[ \t]+$', '', r) for r in white_spaces]
        return white_spaces_end

    @staticmethod
    def clean_emojis(texts: list) -> list:
        return [emoji.replace_emoji(t, replace=' emoji ') for t in texts]

    @staticmethod
    def reduce_repeated_chars(texts: list) -> list:
        # Reduces character repetition like "soooon" -> "soon"
        return [re.sub(r'(.)\1{2,}', r'\1\1', t) for t in texts]

    @staticmethod
    def remove_accents(texts: list) -> list:
        def strip_accents(s):
            return ''.join(
                c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn'
            )
        return [strip_accents(t) for t in texts]

    @staticmethod
    def replace_slang(texts: list) -> list:
        slang_dict = {
            'vc': 'você',
            'pq': 'porque',
            'blz': 'beleza',
            'n': 'não',
            'td': 'tudo',
            'q': 'que',
            'kd': 'cadê',
            'msg': 'mensagem',
            'obg': 'obrigado',
            'vlw': 'valeu',
            'aki': 'aqui',
        }

        def replace(text):
            for key, value in slang_dict.items():
                text = re.sub(rf'\b{key}\b', value, text, flags=re.IGNORECASE)
            return text

        return [replace(t) for t in texts]


class FeatureExtractionUtils:
    """
    Utility class for feature extraction from text data.
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



def plot_sentiment_prediction(input_text, text_pipeline, tfidf_vectorizer, classifier_model):
    """
    Visualizes the sentiment prediction of a given text input using a trained model.

    Parameters
    ----------
    input_text : str or list of str
        Text or list of texts to analyze sentiment.
    text_pipeline : sklearn.Pipeline
        Preprocessing pipeline (e.g., lowercasing, cleaning, etc.).
    tfidf_vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer.
    classifier_model : sklearn classifier
        Trained classification model supporting predict() and predict_proba().
    """
    # Ensure input is a list
    if isinstance(input_text, str):
        input_text = [input_text]

    # Apply preprocessing and vectorization
    preprocessed_texts = text_pipeline.fit_transform(input_text)
    text_features = tfidf_vectorizer.transform(preprocessed_texts)
    
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    X_test = pd.DataFrame(text_features.toarray(), columns=feature_names)
    
    # Predict sentiment and probability
    predictions = classifier_model.predict(X_test)
    probabilities = classifier_model.predict_proba(X_test)

    # # Prepare output for the first item (assumes single input or uses first)
    # label = "Positive" if predictions[0] == 1 else "Negative"
    # confidence = 100 * round(probabilities[0][1 if predictions[0] == 1 else 0], 2)
    # color = "forestgreen" if label == "Positive" else "firebrick"

    # # Plotting the result
    # fig, ax = plt.subplots(figsize=(4, 2))
    # ax.text(0.5, 0.5, label, fontsize=50, ha='center', color=color)
    # ax.text(0.5, 0.0, f"{confidence}%", fontsize=14, ha='center', fontweight='bold')
    # ax.set_title("Sentiment Analysis", fontsize=14)
    # ax.axis('off')
    # plt.show()
    

    # Determina o rótulo e a confiança
    label = "Positive" if predictions[0] == 1 else "Negative"
    confidence = 100 * round(probabilities[0][1 if predictions[0] == 1 else 0], 2)

    color = "green" if label == "Positive" else "red"
    
    fig, ax = plt.subplots(figsize=(3, 1))
    
    ax.set_title(f'Sentiment Analysis: {confidence:.2f}%', fontsize=14, fontweight='bold')
    ax.text(0.5, 0.0, label, fontsize=50, ha='center', color=color)
    # ax.text(0.5, 0.0, f'', fontsize=14, ha='center')
    ax.axis('off')
    
    plt.show()