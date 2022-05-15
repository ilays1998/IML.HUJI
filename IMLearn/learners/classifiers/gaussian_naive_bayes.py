from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        from sklearn.naive_bayes import GaussianNB
        self.priors_ = np.bincount(y.astype(int)) / len(y)
        self.n_classes_ = np.max(y) + 1
        self.classes_ = y
        #self.mu_ = np.array([X[np.where(y == i)].mean(axis=0) for i in range(y.size)])
        #self.vars_ = np.array([X[np.where(y == i)].var(axis=0) for i in range(y.size)])

        res = []
        for i in range(len(X)):
            probas = []
        #    for j in range(y.size):
         #       probas.append((1 / np.sqrt(2 * np.pi * self.vars_[j]) * np.exp(
         #           -0.5 * ((X[i] - self.mu_[j]) / self.vars_[j]))).prod() * self.priors_[j])
            probas = np.array(probas)
            res.append(probas / probas.sum())
        self.k =GaussianNB()
        self.k.fit(X, y)
        self.pi_ = np.array(res)

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
        #return self.pi_.argmax(axis=1)
        return self.k.predict(X)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        liklihood = np.zeros_like(X)
        for i in range(len(X)):
            for j in range(self.n_classes_):
                liklihood[i, j] = ((1 / np.sqrt(2 * np.pi * self.vars_[j]) * np.exp(
                    -0.5 * ((X[i] - self.mu_[j]) / self.vars_[j]))).prod() * self.priors_[j])
        return liklihood

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self._predict(X))

    def pi(self):
        return self.pi_
