from __future__ import annotations
import os, sys, inspect
from typing import NoReturn
import numpy as np
from numpy.linalg import pinv
from IMLearn.metrics import loss_functions
import pandas as pd


class LinearRegression():
    """
    Linear Regression Estimator
    Solving Ordinary Least Squares optimization problem
    """

    def __init__(self, include_intercept: bool = True) -> LinearRegression:
        """
        Instantiate a linear regression estimator
        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not
        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not
        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """
        super().__init__()
        self.include_intercept_, self.coefs_ = include_intercept, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to given samples
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        if self.include_intercept_ == True:
            new_column = np.ones(np.size(X, 0))
            X = np.insert(X, 0, new_column, axis=1)
        X = np.nan_to_num(X)
        B = pinv(X)
        self.coefs_ = B @ y

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for
        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        if self.include_intercept_ == True:
            new_column = np.ones(np.size(X, 0))
            X = np.insert(X, 0, new_column, axis=1)
        X = np.nan_to_num(X)
        return X @ self.coefs_


    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples
        y : ndarray of shape (n_samples, )
            True labels of test samples
        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        if self.include_intercept_ == True:
            new_column = np.ones(np.size(X, 0))
            X = np.insert(X, 0, new_column, axis=1)
        X = np.nan_to_num(X)
        y_hat = X @ self.coefs_
        return loss_functions.mean_square_error(y, y_hat)