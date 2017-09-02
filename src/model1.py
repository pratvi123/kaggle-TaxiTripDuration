#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-8-30, 15:24

@Description:

@Update Date: 17-8-30, 15:24
"""

import numpy as np
import pandas as pd
from util import save_result, preprocess_time
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV


def main():
    test = pd.read_csv("../data/test.csv")
    train = pd.read_csv("../data/train.csv")
    test = preprocess_time(test)
    train = preprocess_time(train)

    # KNN regressor start_end_Lon_lat grid_search


    params = {"n_neighbors": [10, 20, 30], "leaf_size": [30, 50, 70], "p": [1, 2, 3],
              "weights": ["uniform", "distance"]}
    model = GridSearchCV(KNeighborsRegressor(), params, cv=5, verbose=2, scoring="neg_mean_squared_log_error",
                         n_jobs=-1)
    print model

    min_max_scaler = MinMaxScaler()
    train_x = train[["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]].values
    train_y = train["trip_duration"].values
    test_x = test[["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]].values
    min_max_scaler.fit(np.concatenate([train_x, test_x]))
    transform_train_x = min_max_scaler.transform(train_x)
    transform_test_x = min_max_scaler.transform(test_x)
    # model = KNeighborsRegressor()
    model.fit(transform_train_x, train_y)
    print "best_params:", model.best_params_
    print "best_score", model.best_score_
    predicts = model.predict(transform_test_x)
    test["trip_duration"] = predicts
    save_result(test[["id", "trip_duration"]], "../result/start_end_Lon_Lat_KNN_gridsearch_avg.csv")


if __name__ == '__main__':
    main()
