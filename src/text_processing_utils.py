import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import unicodedata
import emoji
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from transformers import AutoTokenizer, AutoModel

import torch



class RegexCleanerTransformer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that performs a series of predefined regex-based 
    text normalization and cleaning operations on a corpus of strings.

    This transformer is intended to be used as part of a preprocessing pipeline 
    for natural language processing (NLP) tasks. It applies a fixed sequence of 
    cleaning steps such as removing URLs, replacing slang, stripping accents, 
    removing special characters, normalizing whitespace, and more.

    Each transformation is applied in sequence to ensure consistent and robust 
    text preprocessing, which is often a crucial step before vectorization or 
    modeling.

    Attributes:
    -----------
    cleaning_steps : list of callable
        A list of text transformation functions applied sequentially.

    Methods:
    --------
    fit(X, y=None)
        Does nothing and returns self, for compatibility with scikit-learn pipelines.

    transform(X, y=None)
        Applies all cleaning functions to the input list of strings.

    Example:
    --------
    pipeline = Pipeline([
        ('regex_cleaner', RegexCleanerTransformer()),
        ('vectorizer', TfidfVectorizer()),
        ('classifier', LogisticRegression())
    ])
    """
    def __init__(self, regex_functions):
        self.regex_functions = regex_functions  # Store the dictionary of regex functions
        # self.regex_functions = [
        #     self.clean_line_breaks,
        #     self.clean_links,
        #     self.clean_dates,
        #     self.clean_currency,
        #     self.clean_numbers,
        #     self.clean_negations,
        #     self.clean_special_characters,
        #     self.clean_whitespace,
        #     self.clean_emojis,
        #     self.reduce_repeated_chars,
        #     self.remove_accents,
        #     self.replace_slang,
        # ]
        
    def fit(self, X, y=None):
        # No fitting necessary, return self for pipeline compatibility
        return self
    
    def transform(self, X, y=None):
        # Apply each regex function sequentially to the input text
        for function in self.regex_functions:
            X = function(X)
        return X




# class RegexCleaner:
    """
    Collection of static methods for regex-based text cleaning.
    Each method receives a list of texts and returns the cleaned list.
    """

    @staticmethod
    def clean_line_breaks(texts: list) -> list:
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


class StopwordRemover(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that removes stopwords from a corpus of text documents.

    This transformer is designed to be used in text preprocessing pipelines. It removes common words 
    (stopwords) from each input document to reduce noise and improve the signal for downstream tasks 
    like vectorization or classification.

    Stopwords are provided at initialization as a custom list, allowing flexibility for different 
    languages or domain-specific vocabularies.

    Attributes:
    -----------
    stopwords : list of str
        A list of stopwords that will be excluded from the text during transformation.

    Methods:
    --------
    fit(X, y=None)
        No fitting is required. Returns self for compatibility with scikit-learn pipelines.

    transform(X, y=None)
        Applies stopword removal to each string in the input list of documents.

    remove_stopwords(text: str) -> list[str]
        Helper method that tokenizes a single string and filters out stopwords.

    Example:
    --------
    from sklearn.pipeline import Pipeline
    from nltk.corpus import stopwords

    pipeline = Pipeline([
        ('stopword_removal', StopwordRemover(stopwords.words('portuguese'))),
        ('vectorizer', TfidfVectorizer())
    ])
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
    A scikit-learn compatible transformer that applies stemming to each word in a corpus of text documents.

    Stemming is a text normalization technique that reduces words to their base or root form. This is useful 
    in natural language processing tasks to treat different forms of a word (e.g., "running", "runs", "ran") 
    as equivalent.

    This transformer uses a customizable stemmer object (such as NLTK's `RSLPStemmer`, `PorterStemmer`, 
    or `SnowballStemmer`), allowing it to support multiple languages and stemming strategies.

    Attributes:
    -----------
    stemmer : object
        A stemmer object with a `.stem()` method. Must be compatible with NLTK-style stemming interfaces.

    Methods:
    --------
    fit(X, y=None)
        Returns self. No fitting is performed; included for compatibility with scikit-learn pipelines.

    transform(X, y=None)
        Applies the stemming function to each word in each input document.

    apply_stemming(text: str) -> list[str]
        Helper function that applies the stemmer to a single string and returns a list of stemmed tokens.

    Example:
    --------
    from nltk.stem import RSLPStemmer
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([
        ('stemmer', StemmerTransformer(RSLPStemmer())),
        ('vectorizer', CountVectorizer())
    ])
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
    A scikit-learn compatible transformer that converts a corpus of text documents 
    into a numeric feature matrix using a provided vectorizer.

    This transformer acts as a wrapper for any scikit-learn-compatible vectorizer, such as 
    `CountVectorizer` or `TfidfVectorizer`. It transforms raw text into a dense numerical array 
    that can be used directly by machine learning models.

    This design allows easy integration into preprocessing pipelines, where the vectorizer 
    can be customized and swapped as needed.

    Attributes:
    -----------
    vectorizer : object
        A scikit-learn vectorizer instance (e.g., TfidfVectorizer, CountVectorizer) that supports
        `fit_transform()` and `toarray()` methods.

    Methods:
    --------
    fit(X, y=None)
        Returns self. The actual fitting is delegated to the internal vectorizer during transform.

    transform(X, y=None)
        Fits and transforms the input text data using the vectorizer, and returns a dense array.

    Example:
    --------
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([
        ('vectorizer', TextVectorizer(TfidfVectorizer())),
        ('classifier', LogisticRegression())
    ])
    """
    
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer 
        
    def fit(self, X, y=None):
        self.vectorizer.fit(X)
        return self
    
    def transform(self, X, y=None):
        return self.vectorizer.transform(X).toarray()


class RegexInspector:
    """
    A utility class for inspecting and comparing regular expression (regex) patterns
    and transformation results in a corpus of text data.

    This class is particularly useful during the development and debugging of
    regex-based preprocessing steps. It helps to identify how and where specific
    regex patterns match within text inputs, and to visualize the effect of
    transformations by comparing the text before and after processing.

    This class is not designed for use in scikit-learn pipelines, but rather as a
    supporting tool during exploratory data analysis or pipeline debugging.

    Methods:
    --------
    find_pattern_spans(pattern: str, texts: list[str]) -> dict
        Searches for all occurrences of a regex pattern in a list of text strings,
        returning the span indices of each match per text.

    print_transformation_comparison(before_texts: list, after_texts: list, indexes: list[int])
        Prints a side-by-side comparison of text data before and after a given
        transformation step. Useful for visual inspection and debugging.

    Example:
    --------
    pattern = r'https?://\S+'
    matches = RegexInspector.find_pattern_spans(pattern, text_samples)

    RegexInspector.print_transformation_comparison(
        before_texts=raw_texts,
        after_texts=cleaned_texts,
        indexes=[0, 5, 12]
    )
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




class BertEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="neuralmind/bert-base-portuguese-cased", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def transform(self, X, y=None):
        embeddings = []
        for text in X:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            embeddings.append(cls_embedding)
        return np.array(embeddings)

    def fit(self, X, y=None):
        return self
    
    
    