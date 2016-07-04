import quandl
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import os
import pandas as pd
import numpy as np


class DataSource:
    quandl.ApiConfig.api_key = "nVDDmKVv_Gd5byZhaCu-"

    labels_X = ['Open', 'High', 'Low', 'Close', 'Volume']
    label_y = ['Adj. Close']

    @classmethod
    def get_data(cls, start_date, end_date, ticker, prediction_horizon=1):
        try:
            filename = "./cache/%s-%s-%s.pickle".format(ticker, start_date, end_date)
            if os.path.isfile(filename):
                data_frame = pd.read_pickle(filename)
            else:
                data_frame = quandl.get("WIKI/" + ticker, start_date=start_date, end_date=end_date)
                data_frame.to_pickle(filename)
            X = data_frame[cls.labels_X][:len(data_frame)-prediction_horizon]
            y = data_frame[cls.label_y][prediction_horizon:]
            return X, y
        except quandl.errors.quandl_error.InvalidRequestError as err:
            print(err)
            print("Please check you have selected a valid time interval")
            return None
        except quandl.errors.quandl_error.NotFoundError as err:
            print("Stock not found. Please specify a valid ticker")
            return None


class Learner:
    def __init__(self):
        raise NotImplementedError()

    def train(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()

    def rmse(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    def score(self, X, y):
        return float(self.rmse(self.predict(X), y))

    @classmethod
    def cross_validation_score(cls, X, y, k=10):
        mean = 0
        n = int(len(y)/k)
        for i in range(1, k):
            X_train = X[:][:i*n]
            y_train = y[:][:i*n]
            X_test = X[:][i*n:(i+1)*n]
            y_test = y[:][i*n:(i+1)*n]
            model = cls()
            model.train(X_train, y_train)
            mean += model.score(X_test, y_test)
        return mean/(k-1)



class LinearRegressionLearner(Learner):
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X, y):
        # TODO normalise
        self.model.fit(X, y)

    def predict(self, X):
        # TODO normalise
        return self.model.predict(X)


class KNNLearner(Learner):
    def __init__(self, k=5):
        self.model = KNeighborsRegressor(n_neighbors=k)

    def train(self, X, y):
        # TODO normalise
        self.model.fit(X, y)

    def predict(self, X):
        # TODO normalise
        return self.model.predict(X)


if __name__ == "__main__":
    start_date = datetime.datetime.strptime('01012015', "%d%m%Y").date()
    end_date = datetime.datetime.strptime('31122015', "%d%m%Y").date()
    X, y = DataSource.get_data(start_date, end_date, "AAPL")

    # linear regression
    linear_regression_learner = LinearRegressionLearner()
    linear_regression_learner.train(X, y)
    print('Linear Regression in-sample error: ', linear_regression_learner.score(X, y))
    print('Linear Regression out-sample error: ', LinearRegressionLearner.cross_validation_score(X, y))

    # KNN
    knn_learner = KNNLearner(k=5)
    knn_learner.train(X, y)
    print('KNN in-sample error: ', knn_learner.score(X, y))
    print('KNN out-sample error: ', KNNLearner.cross_validation_score(X, y))
