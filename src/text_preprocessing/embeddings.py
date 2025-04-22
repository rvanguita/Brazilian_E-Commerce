from sklearn.base import BaseEstimator, TransformerMixin

from transformers import AutoTokenizer, AutoModel

import torch

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
    
    
    