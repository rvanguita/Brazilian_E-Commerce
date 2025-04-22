import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, List, Union


def to_string_list(X: Union[pd.Series, pd.DataFrame, List[str]]) -> List[str]:
    if isinstance(X, pd.DataFrame):
        return X.iloc[:, 0].astype(str).fillna("").tolist()
    if isinstance(X, pd.Series):
        return X.astype(str).fillna("").tolist()
    if isinstance(X, list):
        return ["" if pd.isna(x) else str(x) for x in X]
    raise ValueError("Input must be a list, Series, or DataFrame containing strings.")


class BertEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        model_name: str = "neuralmind/bert-base-portuguese-cased",
        device: Optional[str] = None,
        max_length: int = 256,
        batch_size: int = 8
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size
        self._initialize_model()

    def _initialize_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None) -> np.ndarray:
        print("Starting transformation in BertEmbeddingTransformer...")
        
        texts = to_string_list(X)
        embeddings = []
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc="BERT Encoding"):
            batch_texts = texts[i:i + self.batch_size]
            encoded = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            encoded = {key: val.to(self.device) for key, val in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**encoded)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.extend(cls_embeddings)

        print("Finished transformation in BertEmbeddingTransformer.")
        return np.vstack(embeddings)
