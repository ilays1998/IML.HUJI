#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
from numpy import exp, arange
from pylab import meshgrid, cm, imshow, contour, clabel, colorbar, axis, title, show
from numpy.linalg import inv


class UnivariateGaussian:
    def pdf(l, mu, sigma):
        Y = []
        for i in l:
            Y.append((1 / math.sqrt(2 * sigma)) * math.exp(-pow(i - mu, 2) / (2 * pow(sigma, 2))))
        return Y

    def Question_1():
        print("\nQuestion_1:\n")
        m = 1000
        X = np.random.normal(10, 1, m)
        estimated_expectation = np.mean(X)
        A = X - np.full((1, m), np.mean(X))
        estimated_variance = (1 / (m - 1)) * np.sum((A @ np.transpose(A)))
        print("estimated expectation and estimated variance:\n", (estimated_expectation, estimated_variance))

    def Question_2():
        print("\nQuestion_2:\n")
        func_data = []
        sample_size = np.arange(10, 1001, 10)
        for i in sample_size:
            X = np.random.normal(10, 1, i)
            func_data.append(abs(10 - np.mean(X)))
        go.Figure([go.Scatter(x=sample_size, y=func_data, mode='markers+lines', name=r'$\widehat\|\hat\mu - \mu|$')],
                  layout=go.Layout(title=r"$\text{(1) Estimation of Expectation As Function Of Number Of Samples}$",
                                   xaxis_title="$m\\text{ - number of samples}$",
                                   yaxis_title="r$|\hat\mu - \mu|$",
                                   height=500)).show()

    def Question_3():
        print("\nQuestion_3:\n")
        m = 1000
        X = np.random.normal(10, 1, m)
        X_axis = np.linspace(3, 17, m)
        fig = make_subplots(rows=1, cols=1).add_traces(
            [go.Scatter(x=X, y=UnivariateGaussian.pdf(X, 10, 1), mode='markers', marker=dict(color="blue", size=3))]).update_layout(
            title_text=r"$\text{(2) 1000 Samples with ~}\ N(10, 1) \text{ as a function of the PDF of the distribution}$",
            xaxis_title=r"$\text{sample value}$",
            yaxis_title=r"$\text{PDF of } \ N(10, 1)$",
            height=500, width=1000)
        fig.show()

class Multivariate:
    def log_likelihood(K, mu, sigma, m):
        value = 0
        inverse_sigma = inv(sigma)
        for i in K:
            value += (i - mu) @ inverse_sigma @ np.transpose(i - mu)
        return -value / m

    # the function that I'm going to plot
    def z_func(K, x, y, sigma, m):
        return Multivariate.log_likelihood(K, [x, 0, y, 0], sigma, m)

    def Question_4():
        print("\nQuestion_4:\n")
        m = 1000
        mu = [0, 0, 4, 0]
        sigma = [[1, 0.2, 0, 0.5],
                 [0.2, 2, 0, 0],
                 [0, 0, 1, 0],
                 [0.5, 0, 0, 1]]
        X = np.random.multivariate_normal(mu, sigma, m)
        estimated_expectation = [[0, 0, 0, 0]]
        for i in X:
            estimated_expectation = np.add(estimated_expectation, i)
        estimated_expectation /= m
        print("estimated expectation:\n", estimated_expectation)
        estimated_covariance = np.zeros((4, 4))
        for i in X:
            estimated_covariance += np.transpose(i - estimated_expectation) @ (i - estimated_expectation)
        estimated_covariance /= (m - 1)
        print("estimated covariance:\n", estimated_covariance)

    def Question_5():
        print("\n\nQuestion_5:\n")
        m = 1000
        mu = [0, 0, 4, 0]
        sigma = [[1, 0.2, 0, 0.5],
                 [0.2, 2, 0, 0],
                 [0, 0, 1, 0],
                 [0.5, 0, 0, 1]]
        K = np.random.multivariate_normal(mu, sigma, m)

        x = np.linspace(-10, 10, 200)
        y = np.linspace(-10, 10, 200)
        X, Y = meshgrid(x, y)  # grid of point
        Z = Multivariate.z_func(K, X, Y, sigma, m)  # evaluation of the function on the grid

        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=cm.RdBu, linewidth=0, antialiased=False)

        # ax.zaxis.set_major_locator(LinearLocator(100))
        # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        ax.set_xlabel('--F1--')
        ax.set_ylabel('--F3--')
        ax.set_zlabel('log likelihood')
        ax.set_title('(3.0)log-likelihood as function of F1 and F3')

        plt.show()

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(Y, X, Z, rstride=1, cstride=1,
                               cmap=cm.RdBu, linewidth=0, antialiased=False)

        # ax.zaxis.set_major_locator(LinearLocator(100))
        # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        ax.set_xlabel('--F3--')
        ax.set_ylabel('--F1--')
        ax.set_zlabel('log likelihood')
        ax.set_title('(3.1)log-likelihood as function of F1 and F3 **rotate**')

        plt.show()

    def Question_6():
        m = 1000
        mu = [0, 0, 4, 0]
        sigma = [[1, 0.2, 0, 0.5],
                 [0.2, 2, 0, 0],
                 [0, 0, 1, 0],
                 [0.5, 0, 0, 1]]
        x = np.linspace(-10, 10, 200)
        y = np.linspace(-10, 10, 200)
        K = np.random.multivariate_normal(mu, sigma, m)
        X, Y = meshgrid(x, y)  # grid of point
        Z = Multivariate.z_func(K, X, Y, sigma, m)  # evaluation of the function on the grid
        x_value = (np.argmax(Z) % 200) * (20 / 200) - 10
        y_value = int(np.argmax(Z) / 200) * (20 / 200) - 10
        print("\nQuestion_6:\nthe maximum log-likelihood value achieved in:\n", (round(x_value, 3), round(y_value, 3)))
        print("\nand the maximum log-likelihood value =", Z[np.argmax(Z) % 200, int(np.argmax(Z) / 200)])


p1 = UnivariateGaussian
p1.Question_1()
p1.Question_2()
p1.Question_3()

p2 = Multivariate
p2.Question_4()
p2.Question_5()
p2.Question_6()

