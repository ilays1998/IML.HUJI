import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers.decision_stump import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    loss = []
    for i in range(250):
        k = AdaBoost(wl=None, iterations=i)
        k._fit(train_X, train_y)
        loss.append(k._loss(test_X,test_y))
    print(loss)
    if noise == 0:
        m = 1
    else:
        m = 5

    fig = make_subplots(rows=1, cols=1).add_traces(
        [go.Scatter(x=np.linspace(1, len(loss), len(loss)), y=loss, mode='lines+markers',
                    marker=dict(color="blue", size=3))]).update_layout(
        title_text=r"$\text{(" + str(m) + ") test errors as a function of the number of fitted learners}$",
        xaxis_title=r"$\text{number of fitted learners}$",
        yaxis_title=r"$\text{loss}$",
        height=500, width=1000)
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.tree import DecisionTreeClassifier
    d =DecisionTreeClassifier(max_depth=1)
    k_5 = AdaBoostRegressor(n_estimators=T[0])
    k_5.fit(train_X, train_y)
    y_pred_5 = k_5.predict(test_X)
    k_50 = AdaBoostRegressor(n_estimators=T[1])
    k_50.fit(train_X, train_y)
    y_pred_50 = k_50.predict(test_X)
    k_100 = AdaBoostRegressor(n_estimators=T[0])
    k_100.fit(train_X, train_y)
    y_pred_100 = k_100.predict(test_X)
    k_250 = AdaBoostRegressor(n_estimators=T[0])
    k_250.fit(train_X, train_y)
    y_pred_250 = k_250.predict(test_X)
    symbols = []
    for i in test_y:
        if i == 0:
            symbols.append("circle")
        elif i == 1:
            symbols.append("x")
        else:
            symbols.append("star")
    from IMLearn.metrics import accuracy
    
    size_sample = np.power(1 - np.abs(y_pred_250), 5) * 50

    for i, val in enumerate(y_pred_5):
        if val < 0:
            y_pred_5[i] =-1
        else:
            y_pred_5[i] = 1
    for i, val in enumerate(y_pred_50):
        if val < 0:
            y_pred_50[i] =-1
        else:
            y_pred_50[i] = 1
    for i, val in enumerate(y_pred_100):
        if val < 0:
            y_pred_100[i] =-1
        else:
            y_pred_100[i] = 1
    for i, val in enumerate(y_pred_250):
        if val < 0:
            y_pred_250[i] =-1
        else:
            y_pred_250[i] = 1

    if noise == 0:
        l = 2
    else:
        l = 6

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
        "\nadaboost 5 iteartion - accuracy = " + str(round(accuracy(test_y, y_pred_5), 3)),
        "\nadaboost 50 iteartion - accuracy = " + str(round(accuracy(test_y, y_pred_50), 3)),
        "\nadaboost 100 iteartion - accuracy = " + str(round(accuracy(test_y, y_pred_100), 3)),
        "\nadaboost 250 iteartion - accuracy = " + str(round(accuracy(test_y, y_pred_250), 3))))
    fig.add_traces([decision_surface(k_5.fit(test_X, test_y).predict, lims[0], lims[1], showscale=False),
                    go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(color=y_pred_5, symbol=symbols, line=dict(color="black", width=0.0000001)))],
                   rows=1, cols=1)
    fig.add_traces([decision_surface(k_50.fit(test_X, test_y).predict, lims[0], lims[1], showscale=False),
                    go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(color=y_pred_50, symbol=symbols, line=dict(color="black", width=0.0000001)))],
                   rows=1, cols=2)
    fig.add_traces([decision_surface(k_100.fit(test_X, test_y).predict, lims[0], lims[1], showscale=False),
                    go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(color=y_pred_100, symbol=symbols, line=dict(color="black", width=0.0000001)))],
                   rows=2, cols=1)
    fig.add_traces([decision_surface(k_250.fit(test_X, test_y).predict, lims[0], lims[1], showscale=False),
                    go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(color=y_pred_250, symbol=symbols, line=dict(color="black", width=0.0000001)))],
                   rows=2, cols=2)
    fig.update_layout(
        title_text=r"$\text{---------------------(" + str(l) + ") comparing 5, 50, 100 and 250 iteration with adaboosting ------------\n}$",
        height=600, width=1000)
    fig.show()
    l = l + 1
    if noise == 0:
        title = "the best ensemble = 50 iteration with accuracy " + str(round(accuracy(test_y, y_pred_50), 3)) + \
                " (color = the prediction, shape = the true classification)"
        # Question 3: Decision surface of best performing ensemble
        go.Figure(data=[decision_surface(k_50.fit(test_X, test_y).predict, lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=y_pred_50, symbol=symbols, line=dict(color="black", width=0.0000001)))],
                  layout=go.Layout(title= rf"$\textbf{{({l}) {title} }}$")).show()
    else:
        title = "the best ensemble = 250 iteration with accuracy " + str(round(accuracy(test_y, y_pred_250), 3)) + \
                " (color = the prediction, shape = the true classification)"
        # Question 3: Decision surface of best performing ensemble
        go.Figure(data=[decision_surface(k_250.fit(test_X, test_y).predict, lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=y_pred_250, symbol=symbols,
                                               line=dict(color="black", width=0.0000001)))],
                  layout=go.Layout(title=rf"$\textbf{{({l}) {title} }}$")).show()

    l = l + 1
    title = "the ensemble with point size proportional to it’s weight (color = the prediction," \
            " shape = the true classification)"
    # Question 4: Decision surface with weighted samples
    go.Figure(data=[decision_surface(k_250.fit(test_X, test_y).predict, lims[0], lims[1], showscale=False),
                    go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(color=y_pred_250, symbol=symbols, size=size_sample,
                                           line=dict(color="black", width=0.0000001)))],
              layout=go.Layout(title=rf"$\textbf{{({l}) {title} }}$")).show()



if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
