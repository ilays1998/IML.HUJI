from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)

def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        data = np.load("..\\datasets\\" + f)
        X = data[: , 0:2]
        y = data[: , 2:]
        y = y.ravel()

        if f == "linearly_separable.npy":
            k = 1
        else:
            k = 2

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        percep = Perceptron()
        percep._fit(X, y)
        losses = percep.callback_


        # Plot figure of loss as function of fitting iteration

        fig = make_subplots(rows=1, cols=1).add_traces(
            [go.Scatter(x=np.linspace(1, len(losses), len(losses)), y=losses, mode='lines+markers',
                        marker=dict(color="blue", size=3))]).update_layout(
            title_text=r"$\text{(" + str(k) + ") Perceptron algorithm: the loss of the algorithm as a function of iterations in " + n + " dataset}$",
            xaxis_title=r"$\text{number of iterations}$",
            yaxis_title=r"$\text{loss}$",
            height=500, width=1000)
        fig.show()

def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        data = np.load("..\\datasets\\" + f)
        X = data[:, 0:2]
        y = data[:, 2:]
        y = y.ravel()


        # Fit models and predict over training set
        lda = LDA()
        lda._fit(X, y)
        GAB = GaussianNaiveBayes()
        GAB._fit(X,y)

        lda_pre = lda._predict(X)
        GAB_pre = GAB._predict(X)




        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        symbols = []
        for i in y:
            if i == 0:
                symbols.append("circle")
            elif i == 1:
                symbols.append("x")
            else:
                symbols.append("star")
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("\nClassification with Gaussian Naive Bayes - accuracy = " + str(round(accuracy(y,GAB_pre), 3)),
                            "\nClassification with Linear Discrimination Analysis - accuracy = " + str(round(accuracy(y,lda_pre), 3))))
        lims = np.array([X.min(axis=0), X.max(axis=0)]).T + np.array([-.4, .4])
        fig.add_traces([decision_surface(GAB.fit(X, y).predict, lims[0], lims[1], showscale=False),
                        go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=GAB_pre, symbol=symbols, line=dict(color="black", width=1)))],
                        rows=1, cols=1)
        fig.add_traces([decision_surface(lda.fit(X, y).predict, lims[0], lims[1], showscale=False),
                        go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=lda_pre, symbol=symbols, line=dict(color="black", width=1)))],
                        rows=1, cols=2)
        fig.update_layout(title_text=r"$\text{-------------------------------(3) classification with LDA and GNB  -  Generating Data From " + f + " dataset-------------------\n}$", height=300)


        # Add traces for data-points setting symbols and colors


        # Add `X` dots specifying fitted Gaussians' means
        circle_ = []
        x_ = []
        star_ = []
        for i, val in enumerate(y):
            if val == 0:
                circle_.append(X[i, :])
            if val == 1:
                x_.append(X[i, :])
            if val == 2:
                star_.append(X[i, :])
        star_ = np.array(star_)
        circle_ = np.array(circle_)
        x_ = np.array(x_)

        circle_mean = np.mean(circle_, axis=0)
        star_mean = np.mean(star_, axis=0)
        x_mean = np.mean(x_, axis=0)

        #trace1 = go.Figure(go.Scatter(x=[circle_mean[0], circle_mean[0] - 2, circle_mean[0] + 2],y=[circle_mean[1], circle_mean[1] - 2, circle_mean[1] + 2], fill="toself"))

        fig.add_trace(go.Scatter(x=[circle_mean[0], circle_mean[0] - 1, circle_mean[0] + 1], y=[circle_mean[1], circle_mean[1] + 1, circle_mean[1] - 1],line=dict(color="yellow", width=1),showlegend=False),row=1, col=1)
        fig.add_trace(go.Scatter(x=[circle_mean[0], circle_mean[0] - 1, circle_mean[0] + 1], y=[circle_mean[1], circle_mean[1] - 1, circle_mean[1] + 1],line=dict(color="yellow", width=1), showlegend=False),row=1, col=1)
        fig.add_trace(go.Scatter(x=[circle_mean[0], circle_mean[0] - 1, circle_mean[0] + 1], y=[circle_mean[1], circle_mean[1] - 1, circle_mean[1] + 1],line=dict(color="yellow", width=1),showlegend=False),row=1, col=2)
        fig.add_trace(go.Scatter(x=[circle_mean[0], circle_mean[0] - 1, circle_mean[0] + 1], y=[circle_mean[1], circle_mean[1] + 1, circle_mean[1] - 1],line=dict(color="yellow", width=1),showlegend=False),row=1, col=2)
        fig.add_trace(go.Scatter(x=[star_mean[0], star_mean[0] - 1, star_mean[0] + 1], y=[star_mean[1], star_mean[1] - 1, star_mean[1] + 1],line=dict(color="pink", width=1),showlegend=False),row=1, col=1)
        fig.add_trace(go.Scatter(x=[star_mean[0], star_mean[0] - 1, star_mean[0] + 1], y=[star_mean[1], star_mean[1] + 1, star_mean[1] - 1],line=dict(color="pink", width=1),showlegend=False),row=1, col=1)
        fig.add_trace(go.Scatter(x=[star_mean[0], star_mean[0] - 1, star_mean[0] + 1], y=[star_mean[1], star_mean[1] - 1, star_mean[1] + 1],line=dict(color="pink", width=1),showlegend=False),row=1, col=2)
        fig.add_trace(go.Scatter(x=[star_mean[0], star_mean[0] - 1, star_mean[0] + 1], y=[star_mean[1], star_mean[1] + 1, star_mean[1] - 1],line=dict(color="pink", width=1),showlegend=False),row=1, col=2)
        fig.add_trace(go.Scatter(x=[x_mean[0], x_mean[0] - 1, x_mean[0] + 1], y=[x_mean[1], x_mean[1] - 1, x_mean[1] + 1],line=dict(color="red", width=1),showlegend=False),row=1, col=1)
        fig.add_trace(go.Scatter(x=[x_mean[0], x_mean[0] - 1, x_mean[0] + 1], y=[x_mean[1], x_mean[1] + 1, x_mean[1] - 1],line=dict(color="red", width=1),showlegend=False),row=1, col=1)
        fig.add_trace(go.Scatter(x=[x_mean[0], x_mean[0] - 1, x_mean[0] + 1], y=[x_mean[1], x_mean[1] - 1, x_mean[1] + 1],line=dict(color="red", width=1),showlegend=False),row=1, col=2)
        fig.add_trace(go.Scatter(x=[x_mean[0], x_mean[0] - 1, x_mean[0] + 1], y=[x_mean[1], x_mean[1] + 1, x_mean[1] - 1],line=dict(color="red", width=1),showlegend=False),row=1, col=2)
        # Add ellipses depicting the covariances of the fitted Gaussians
        fig.add_trace(get_ellipse(circle_mean, np.cov(circle_.T)),row=1, col=2)
        fig.add_trace(get_ellipse(circle_mean, np.cov(circle_.T)),row=1, col=1)
        fig.add_trace(get_ellipse(star_mean, np.cov(star_.T)),row=1, col=2)
        fig.add_trace(get_ellipse(star_mean, np.cov(star_.T)),row=1, col=1)
        fig.add_trace(get_ellipse(x_mean, np.cov(x_.T)),row=1, col=2)
        fig.add_trace(get_ellipse(x_mean, np.cov(x_.T)),row=1, col=1)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
