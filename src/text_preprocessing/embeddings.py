import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, List, Union


def convert_to_string_list(
    input_data: Union[pd.Series, pd.DataFrame, List[str]]
) -> List[str]:
    if isinstance(input_data, pd.DataFrame):
        return input_data.iloc[:, 0].astype(str).fillna("").tolist()
    if isinstance(input_data, pd.Series):
        return input_data.astype(str).fillna("").tolist()
    if isinstance(input_data, list):
        return ["" if pd.isna(item) else str(item) for item in input_data]
    raise ValueError("Input must be a list, Series, or DataFrame of strings.")


class BertEmbeddingEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        model_name: str = "neuralmind/bert-base-portuguese-cased",
        device: Optional[str] = None,
        max_length: int = 256,
        batch_size: int = 8,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size
        self._load_model()

    def _load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None) -> np.ndarray:
        texts = convert_to_string_list(X)
        all_embeddings = []

        for start_idx in tqdm(
            range(0, len(texts), self.batch_size), desc="Generating BERT embeddings"
        ):
            batch = texts[start_idx : start_idx + self.batch_size]
            tokenized = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

            with torch.no_grad():
                model_output = self.model(**tokenized)
            cls_embedding = model_output.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embedding)

        return np.vstack(all_embeddings)
