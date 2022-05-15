import numpy as np


from ..base import BaseEstimator
from typing import Callable, NoReturn
from IMLearn.learners.classifiers.decision_stump import DecisionStump
import pandas as pd

class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.alphas = []
        self.G_M = []
        self.training_errors = []
        self.prediction_errors = []
        self.models_, self.weights_, self.D_ = None, None, None

    def compute_error(self, y, y_pred, w_i):
        '''
        Calculate the error rate of a weak classifier m. Arguments:
        y: actual target value
        y_pred: predicted value by weak classifier
        w_i: individual weights for each observation

        Note that all arrays should be the same length
        '''
        return (sum(w_i * (np.not_equal(y, y_pred)).astype(int))) / sum(w_i)

    def compute_alpha(self, error):
        '''
        Calculate the weight of a weak classifier m in the majority vote of the final classifier. This is called
        alpha in chapter 10.1 of The Elements of Statistical Learning. Arguments:
        error: error rate from weak classifier m
        '''
        return np.log((1 - error) / error)

    def update_weights(self, w_i, alpha, y, y_pred):
        '''
        Update individual weights w_i after a boosting iteration. Arguments:
        w_i: individual weights for each observation
        y: actual target value
        y_pred: predicted value by weak classifier
        alpha: weight of weak classifier used to estimate y_pred
        '''
        return w_i * np.exp(alpha * (np.not_equal(y, y_pred)).astype(int))
    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        global w_i, alpha_m, y_pred
        self.alphas = []
        self.training_errors = []
        self.M = self.iterations_

        # Iterate over M weak classifiers
        for m in range(0, self.iterations_):

            # Set weights for current boosting iteration
            if m == 0:
                w_i = np.ones(len(y)) * 1 / len(y)  # At m = 0, weights are all the same and equal to 1 / N
            else:
                # (d) Update w_i
                w_i = self.update_weights(w_i, alpha_m, y, y_pred)

            # (a) Fit weak classifier and predict labels
            from sklearn.tree import DecisionTreeClassifier
            G_m = DecisionTreeClassifier(max_depth=1)  # Stump: Two terminal-node classification tree
            # Stump: Two terminal-node classification tree
            G_m.fit(X, y, sample_weight=w_i)
            y_pred = G_m.predict(X)

            self.G_M.append(G_m)  # Save to list of weak classifiers

            # (b) Compute error
            error_m = self.compute_error(y, y_pred, w_i)
            self.training_errors.append(error_m)

            # (c) Compute alpha
            alpha_m = self.compute_alpha(error_m)
            self.alphas.append(alpha_m)

        assert len(self.G_M) == len(self.alphas)


    def _predict(self, X):
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
        weak_preds = pd.DataFrame(index=range(len(X)), columns=range(self.M))

        # Predict class label for each weak classifier, weighted by alpha_m
        for m in range(self.M):
            y_pred_m = self.G_M[m].predict(X) * self.alphas[m]
            weak_preds.iloc[:, m] = y_pred_m

        # Calculate final predictions
        y_pred = (1 * np.sign(weak_preds.T.sum())).astype(int)
        return y_pred

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
        from ..metrics import misclassification_error
        return misclassification_error(y, self._predict(X))

    def accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate accuracy of given predictions

        Parameters
        ----------
        y_true: ndarray of shape (n_samples, )
            True response values
        y_pred: ndarray of shape (n_samples, )
            Predicted response values

        Returns
        -------
        Accuracy of given predictions
        """
        sum = 0
        for i, val in enumerate(y_true):
            if int(val) == int(y_pred[i]):
                sum += 1
        return sum / y_true.size

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        weak_preds = pd.DataFrame(index=range(len(X)), columns=range(T))

        # Predict class label for each weak classifier, weighted by alpha_m
        for m in range(T):
            y_pred_m = self.G_M[m].predict(X) * self.alphas[m]
            weak_preds.iloc[:, m] = y_pred_m

        # Calculate final predictions
        y_pred = (1 * np.sign(weak_preds.T.sum())).astype(int)
        return y_pred


    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ..metrics import misclassification_error
        return misclassification_error(y, self.partial_predict(X, T))