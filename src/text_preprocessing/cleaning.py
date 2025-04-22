import unicodedata
import emoji
import re


from sklearn.base import BaseEstimator, TransformerMixin




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
