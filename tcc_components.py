import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class TextConcatenator(BaseEstimator, TransformerMixin):
    """Concatena colunas de texto em uma unica string por linha."""
    def __init__(self, text_cols=("review_title","review_text")):
        self.text_cols = tuple(text_cols)
    def fit(self, X, y=None): return self
    def transform(self, X):
        df = pd.DataFrame(X).copy()
        missing = [c for c in self.text_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Colunas de texto esperadas nao foram encontradas: {missing}. Available: {list(df.columns)}")
        joined = (df[list(self.text_cols)].fillna("").astype(str).agg(" \n ".join, axis=1))
        return joined

class SimpleCleaner(BaseEstimator, TransformerMixin):
    """Converte para minusculas e suprime espacos."""
    def __init__(self):
        self._ws = re.compile(r"\s+")
    def fit(self, X, y=None): return self
    def transform(self, X):
        out = []
        for t in X:
            t = "" if t is None else str(t).lower()
            t = self._ws.sub(" ", t).strip()
            out.append(t)
        return out
