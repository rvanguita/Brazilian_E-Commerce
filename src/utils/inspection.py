import re
import pandas as pd
from typing import List, Dict, Tuple, Union


class RegexInspector:
    """
    Utility class for inspecting regex patterns and comparing text transformations.
    
    Not intended for use in production pipelines, but useful for EDA and debugging.
    """

    @staticmethod
    def find_pattern_spans(pattern: str, texts: List[str]) -> Dict[str, List[Tuple[int, int]]]:
        """
        Finds all regex match spans in a list of text samples.

        Parameters:
        - pattern: Regex pattern as string.
        - texts: List of strings to inspect.

        Returns:
        - Dictionary mapping text indices to match span tuples.
        """
        compiled = re.compile(pattern)
        results = {
            f'Text idx {i}': [match.span() for match in compiled.finditer(text)]
            for i, text in enumerate(texts)
            if compiled.search(text)
        }
        return results

    @staticmethod
    def print_transformation_comparison(
        before_texts: Union[List[str], pd.Series],
        after_texts: Union[List[str], pd.Series],
        indexes: List[int]
    ) -> None:
        """
        Prints before/after comparison of text transformations.

        Parameters:
        - before_texts: Raw text before processing.
        - after_texts: Text after transformation.
        - indexes: List of indices to display.
        """
        for i, idx in enumerate(indexes, start=1):
            before = before_texts[idx] if isinstance(before_texts, list) else before_texts.iloc[idx]
            after = after_texts[idx] if isinstance(after_texts, list) else after_texts.iloc[idx]
            print(f'--- Text {i} (Index {idx}) ---\n')
            print(f'Before:\n{before}\n')
            print(f'After:\n{after}\n')
            print('-' * 60)
