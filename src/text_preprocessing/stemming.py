from sklearn.base import BaseEstimator, TransformerMixin


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

