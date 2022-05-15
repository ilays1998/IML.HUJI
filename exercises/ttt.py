if __name__ == '__main__':
    from sklearn.naive_bayes import GaussianNB
    import numpy as np

    GAB = GaussianNB()
    X = [0, 1, 2 ,3 ,4 ,5 ,6 ,7]
    X = np.array(X)
    X = X.reshape(-1, 1)
    print(X)
    y = [0, 0, 1, 1, 1, 1, 2, 2]
    GAB.fit(X, y)
    print(X[:1, :])
    print(GAB.predict_proba([[2]]))
    print(GAB.get_params())
