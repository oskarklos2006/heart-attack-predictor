import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        Q1 = pd.DataFrame(X).quantile(0.25)
        Q3 = pd.DataFrame(X).quantile(0.75)
        IQR = Q3 - Q1
        self.lower_ = Q1 - self.factor * IQR
        self.upper_ = Q3 + self.factor * IQR
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        return df.clip(lower=self.lower_, upper=self.upper_, axis=1).values
