import quandl
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import os
import pandas as pd
import numpy as np


class DataSource:
    quandl.ApiConfig.api_key = "nVDDmKVv_Gd5byZhaCu-"
    horizons = [1, 7, 14, 28]
    labels = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj. Close']
    label_to_predict = ['Adj. Close']

    @classmethod
    def get_data(cls, start_date, end_date, tickers):
        try:
            data_frames = {}
            for ticker in tickers:
                filename = "./cache/{}-{}-{}.pickle".format(ticker, start_date, end_date)
                if os.path.isfile(filename):
                    data_frame = pd.read_pickle(filename)
                else:
                    data_frame = quandl.get("WIKI/" + ticker, start_date=start_date, end_date=end_date)
                    data_frame.to_pickle(filename)
                data_frames[ticker] = data_frame[cls.labels]
            return cls._get_X_y(data_frames)
        except quandl.errors.quandl_error.InvalidRequestError as err:
            print(err)
            print("Please check you have selected a valid time interval")
            return None
        except quandl.errors.quandl_error.NotFoundError as err:
            print("Stock not found. Please specify a valid ticker")
            return None

    @classmethod
    def _get_X_y(cls, data):
        df = pd.concat(data.values(), axis=1, keys=list(data.keys()))
        df = pd.concat([df.shift(-i) for i in [0] + cls.horizons], axis=1, keys=[0] + cls.horizons).dropna()
        X = df[0]
        y = pd.concat([df[i] for i in cls.horizons], axis=1, keys=cls.horizons)
        return X, y


class Learner:
    def __init__(self):
        raise NotImplementedError

    def train(self, X, y):
        self.model.fit(X, y)

    def forecast(self, X, y, days):
        last = X.tail(1)
        remaining = days
        while remaining > 0:
            for h in DataSource.horizons[::-1]:
                if remaining - h >= 0:
                    prediction = pd.DataFrame(self.model.predict(X), columns=y.columns)
                    last = prediction[h]
                    remaining -= h
                    break
        return last

    def predict(self, X, y):
        return pd.DataFrame(self.model.predict(X), columns=y.columns, index=y.index)

    def _rmse(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    def score(self, X, y):
        return self._rmse(self.predict(X, y), y)


class LinearRegressionLearner(Learner):
    def __init__(self):
        self.model = Pipeline(steps=[('features_scaling', MinMaxScaler()), ('linear_regression', LinearRegression())])


class KNNLearner(Learner):
    def __init__(self, k=5):
        self.model = Pipeline(steps=[('features_scaling', MinMaxScaler()), ('knn', KNeighborsRegressor(n_neighbors=k))])


class LearnerEvaluator:
    @classmethod
    def cross_validation_score(cls, learner_class, start_date, end_date, tickers, k=10):
        X, y = DataSource.get_data(start_date, end_date, tickers)
        mean_in_sample = 0
        mean_out_sample = 0
        n = int(len(y) / k)
        for i in range(1, k):
            X_train = X[:][:i * n]
            y_train = y[:][:i * n]
            X_test = X[:][i * n:(i + 1) * n]
            y_test = y[:][i * n:(i + 1) * n]
            learner = learner_class()
            learner.train(X, y)
            mean_in_sample += learner.score(X_train, y_train)
            mean_out_sample += learner.score(X_test, y_test)
            score_in_sample = mean_in_sample[:, :, DataSource.label_to_predict[0]] / (k - 1)
            score_out_sample = mean_out_sample[:, :, DataSource.label_to_predict[0]] / (k - 1)
        return score_in_sample, score_out_sample


if __name__ == "__main__":
    start_date = datetime.datetime.strptime('01012015', "%d%m%Y").date()
    end_date = datetime.datetime.strptime('31122015', "%d%m%Y").date()
    tickers = ["AAPL", "GOOG"]

    X, y = DataSource.get_data(start_date, end_date, tickers)

    # linear regression
    lin_reg_score_in_sample, lin_reg_score_out_sample = LearnerEvaluator.cross_validation_score(LinearRegressionLearner, start_date, end_date, tickers)
    print("--- Linear Regression ---")
    print("RMSE in sample: ")
    print(lin_reg_score_in_sample)
    print("RMSE out sample: ")
    print(lin_reg_score_out_sample)

    # knn
    knn_score_in_sample, knn_score_out_sample = LearnerEvaluator.cross_validation_score(KNNLearner, start_date, end_date, tickers)
    print("---  KNN ---")
    print("RMSE in sample: ")
    print(knn_score_in_sample)
    print("RMSE out sample: ")
    print(knn_score_out_sample)
