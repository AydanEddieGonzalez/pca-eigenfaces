import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]

        # 1. Center
        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_

        # 2. Covariance matrix (d x d)
        C = X_c.T @ X_c / (n - 1)

        # 3. Eigendecomposition (C is symmetric)
        eigvals, eigvecs = np.linalg.eigh(C)

        # 4. Sort largest first
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # 5. Keep the top k, stored as ROWS
        k = self.n_components
        self.components_ = eigvecs[:, :k].T
        self.explained_variance_ = eigvals[:k]
        self.explained_variance_ratio_ = self.explained_variance_ / eigvals.sum()
        return self

    def transform(self, X):
        X_c = np.asarray(X, dtype=float) - self.mean_
        return X_c @ self.components_.T

    def inverse_transform(self, Z):
        return Z @ self.components_ + self.mean_

    def fit_transform(self, X):
        return self.fit(X).transform(X)