from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset
    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    a = pd.read_csv(filename)
    a.head()
    df = pd.DataFrame(a)
    y = df.pop("price")
    del df["waterfront"]
    del df["view"]
    del df["lat"]
    del df["long"]
    del df["yr_renovated"]
    del df["date"]


    del df["id"]



    return (df, y)




def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = "result") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem
    y : array-like of shape (n_samples, )
        Response vector to evaluate against
    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    #blankIndex = [''] * len(X)
    #X.index = blankIndex
    X["response"] = y
    cov = X.cov()
    for column in X:
        pearson = cov["response"][column] / (np.std(y) * np.std(X[column]))
        fig = make_subplots(rows=1, cols=1).add_traces(
            [go.Scatter(x=X[column], y=y, mode='markers',
                        marker=dict(color="blue", size=3))]).update_layout(
            title_text=r"$\text{correlation between " + column +" and response\n" + "   pearson correlation = " + str(pearson) + "}$",
            xaxis_title=r"$\text{" + column + "}$",
            yaxis_title=r"$\text{response}$",
            height=500, width=1000)
        pio.write_image(fig, "result/" + column + ".jpeg")



if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    print(load_data("house_prices.csv"))

    # Question 2 - Feature evaluation with respect to response
    X, y = load_data("house_prices.csv")
    feature_evaluation(X, y)

    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_x, test_y = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    result = []
    for p in range(10, 100, 10):
        train_x_p, train_y_p, m ,n = split_train_test(train_x, train_y, p / 100)
        linear_reg = LinearRegression()
        linear_reg._fit(train_x_p.to_numpy(), train_y_p.to_numpy())

        test_x_p, test_y_p, a, b = split_train_test(test_x, test_y, p / 100)
        result.append(linear_reg._predict(test_x_p.to_numpy()))
        print(linear_reg._loss(test_x_p.to_numpy(), test_y_p.to_numpy()))

