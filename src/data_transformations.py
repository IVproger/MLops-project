import hashlib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FunctionTransformer


class feature_extractor(BaseEstimator, TransformerMixin):
    def __init__(self, features: list[str]):
        super().__init__()
        self.features = features

    def transform(self, X):
        columns_to_drop = set(X) - set(self.features)

        X_transformed = X[list[columns_to_drop]]

        return X_transformed


class sin_transformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature: str, period: int):
        super().__init__()
        self.feature = feature
        self.period = period

    def transform(self, X):
        func_trans = FunctionTransformer(lambda x: np.sin(x / self.period * 2 * np.pi))
        X[f"{self.feature}_sin"] = func_trans.transform(X[self.feature])

        return X


class cos_transformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature: str, period: int):
        super().__init__()
        self.feature = feature
        self.period = period

    def transform(self, X):
        func_trans = FunctionTransformer(lambda x: np.sin(x / self.period * 2 * np.pi))
        X[f"{self.feature}_cos"] = func_trans.transform(X[self.feature])

        return X


class cyclic_transformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature: str, period: int):
        super().__init__()
        self.feature = feature
        self.period = period

    def transform(self, X):
        X = sin_transformer(
            self.feature,
            self.period,
        ).transform(X)
        X = cos_transformer(
            self.feature,
            self.period,
        ).transform(X)

        return X


class hhmm2hh(BaseEstimator, TransformerMixin):
    def __init__(self, feature: str):
        super().__init__()
        self.column = feature

    def transform(self, X):
        func_trans = FunctionTransformer(lambda x: x // 100)
        X[f"{self.column}_HH"] = func_trans.transform(X[self.column])

        return X


class hhmm2mm(BaseEstimator, TransformerMixin):
    def __init__(self, feature: str):
        super().__init__()
        self.column = feature

    def transform(self, X):
        func_trans = FunctionTransformer(lambda x: x % 100)
        X[f"{self.column}_MM"] = func_trans.transform(X[self.column])
        return X


class hhmm_transformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature: str):
        super().__init__()
        self.feature = feature

    def transform(self, X):
        X = hhmm2hh(self.feature).transform(X)
        X = hhmm2mm(self.feature).transform(X)
        X = cyclic_transformer(f"{self.feature}_HH", 24).transform(X)
        X = cyclic_transformer(f"{self.feature}_MM", 60).transform(X)

        return X


class hash_transformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature: str, buckets=1000):
        super().__init__()
        self.feature = feature
        self.buckets = buckets

    def transform(self, X):
        func_trans = FunctionTransformer(
            lambda series: series.apply(
                lambda text: int(hashlib.md5(text.encode()).hexdigest(), 16)
                % self.buckets
            )
        )
        X[self.feature] = func_trans.transform(X[self.feature])
        return X
