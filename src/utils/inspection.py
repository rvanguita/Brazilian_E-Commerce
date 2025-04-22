import re

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

