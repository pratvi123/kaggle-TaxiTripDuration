#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-8-31, 14:14

@Description:

@Update Date: 17-8-31, 14:14
"""

from util import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
import xgboost as xgb

# 主要是树模型

x_columns = ["vendor_id",
             "passenger_count",
             "pickup_longitude", "pickup_latitude", 'dropoff_longitude', 'dropoff_latitude',
             'store_and_fwd_int',
             'total_distance',
             'number_of_steps',
             'n_step_other',
             'left', 'right', 'straight',
             'uturn', 'slight_left', 'slight_right',
             'sum_left_right_straight', 'sum_all', 'sum_all_is_zero',
             'ratio_left_sum_all', 'ratio_right_sum_all', 'ratio_straight_sum_all',
             'ratio_right_left_sum_all', 'ratio_straight_right_sum_all', 'ratio_left_straight_sum_all',
             'ratio_uturn_sum_all', 'ratio_slight_left_sum_all', 'ratio_slight_right_sum_all',
             'month',
             'hour',
             'weekday',
             "is_weekend",
             'dayofyear',
             'weekofyear',
             'haversine_distance',
             'manhattan_distance',
             'bearing',
             'avg_trip_duration_by_hour_vendor_id',
             'avg_total_travel_time_by_hour_vendor_id',
             'difference_by_hour_vendor_id',
             'ratio_by_hour_vendor_id',
             'ratio_actual_by_hour_vendor_id',
             'avg_trip_duration_by_hour',
             'avg_total_travel_time_by_hour',
             'difference_by_hour',
             'ratio_by_hour',
             'ratio_actual_by_hour',
             "peek_degree",
             'pickup_label', 'dropoff_label',
             'pickup_label_longitude', 'pickup_label_latitude', 'dropoff_label_longitude', 'dropoff_label_latitude',
             'haversine_distance_pcenter_dcenter', 'manhattan_distance_pcenter_dcenter', 'bearing_pcenter_dcenter',
             'haversine_distance_d_dcenter', 'manhattan_distance_d_dcenter', 'bearing_d_dcenter',
             'max_T', 'min_T', 'avg_T', 'precipitation', 'snow_fall', 'snow_depth',
             # "pickup_pca0", "pickup_pca1", "dropoff_pca0", "dropoff_pca1"
             ]


def ExtraTreeMain(train, test):
    print "ExtraTreeMain"
    train_y = train["trip_duration"].values
    train_y = np.log(train_y + 1)
    print "features:", x_columns
    print "feature size:", len(x_columns)
    train_x = train[x_columns]
    test_x = test[x_columns]
    n_estimators = 500
    min_samples_leaf = 20
    model = ExtraTreesRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, n_jobs=-1)
    params = {
        "n_estimators": [200, 500, 100],
        "max_features": ["auto"],
        "max_depth": [None],
        "min_samples_leaf": [10, 20, 30]
    }
    model = train_model(model, train_x, train_y, cv=0, re_fit=True, grid_search=False, grid_params=params)
    predict_and_save_result(model, test_x, test,
                            "../result/ExtraTrees_{}estimators_{}minleaf_{}f.csv".format(n_estimators,
                                                                                         min_samples_leaf,
                                                                                         len(x_columns)),
                            transform="log")


def RandomForestMain(train, test):
    print "RandomForestMain"
    train_y = train["trip_duration"].values
    train_y = np.log(train_y + 1)
    print "features:", x_columns
    print "feature size:", len(x_columns)
    train_x = train[x_columns]
    test_x = test[x_columns]

    min_samples_leaf = 20
    n_estimators = 500
    model = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, n_jobs=-1)
    params = {
        "n_estimators": [200, 500, 1000],
        "max_features": ["auto"],
        "max_depth": [None],
        "min_samples_leaf": [5, 10, 20]
    }
    model = train_model(model, train_x, train_y, cv=0, re_fit=True, grid_search=False, grid_params=params)
    predict_and_save_result(model, test_x, test,
                            "../result/RandomForest_{}esti_{}leaf_{}f.csv".format(n_estimators,
                                                                                  min_samples_leaf,
                                                                                  len(x_columns)),
                            transform="log")


def XGB_Main(train, test):
    print "XGB_Main"
    train_y = train["trip_duration"].values
    train_y = np.log(train_y + 1)
    print "features:", x_columns
    print "feature size:", len(x_columns)
    train_x = train[x_columns].values
    test_x = test[x_columns].values

    start = time.time()
    Xtr, Xv, ytr, yv = train_test_split(train_x, train_y, test_size=0.3, random_state=2017)
    dtrain = xgb.DMatrix(Xtr, label=ytr)
    dvalid = xgb.DMatrix(Xv, label=yv)
    dtest = xgb.DMatrix(test_x)
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

    # Try different parameters! My favorite is random search :)
    lr = 0.2
    xgb_pars = {'min_child_weight': 50, 'eta': lr, 'colsample_bytree': 0.3, 'max_depth': 15,
                'subsample': 0.8, 'lambda': 1., 'nthread': -1, 'booster': 'gbtree', 'silent': 1,
                'eval_metric': 'rmse', 'objective': 'reg:linear'}

    # You could try to train with more epoch
    n_rounds = 100
    model = xgb.train(xgb_pars, dtrain, n_rounds, watchlist, early_stopping_rounds=2,
                      maximize=False, verbose_eval=1)
    print("Time taken by above cell is {}.".format(time.time() - start))
    print('Modeling RMSLE %.5f' % model.best_score)

    predicts = model.predict(dtest, ntree_limit=model.best_ntree_limit)
    predicts = np.exp(predicts) - 1
    test["trip_duration"] = predicts
    save_path = "../result/XGB_{}rounds_{}lr_{}f.csv".format(n_rounds, lr, len(x_columns))
    save_result(test[["id", "trip_duration"]], save_path)


if __name__ == '__main__':
    train, test = load_train_and_test()

    # PCA
    # from sklearn.decomposition import PCA
    #
    # tart = time.time()
    # coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
    #                     train[['dropoff_latitude', 'dropoff_longitude']].values,
    #                     test[['pickup_latitude', 'pickup_longitude']].values,
    #                     test[['dropoff_latitude', 'dropoff_longitude']].values))
    # print coords.shape
    #
    # pca = PCA().fit(coords)
    # train['pickup_pca0'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 0]
    # train['pickup_pca1'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 1]
    # train['dropoff_pca0'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
    # train['dropoff_pca1'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
    # test['pickup_pca0'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 0]
    # test['pickup_pca1'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 1]
    # test['dropoff_pca0'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
    # test['dropoff_pca1'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
    # print train.columns

    # ExtraTreeMain(train, test)
    RandomForestMain(train, test)
    # XGB_Main(train, test)
