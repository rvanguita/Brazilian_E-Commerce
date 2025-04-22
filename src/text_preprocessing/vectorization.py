from sklearn.base import BaseEstimator, TransformerMixin



class TextVectorizerWrapper(BaseEstimator, TransformerMixin):
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
