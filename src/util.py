#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-8-30, 15:24

@Description:

@Update Date: 17-8-30, 15:24
"""

import pandas as pd
from collections import Counter
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.cluster import KMeans
import time
import os

# __amplification_factor = 10 ** 5
__amplification_factor = 1


def train_model(model, trainx, trainy, cv=0, grid_search=False, re_fit=True, grid_params=None):
    start = time.time()
    if not grid_search:
        if cv <= 0:
            model.fit(trainx, trainy)
            print "fit spend ", (time.time() - start)
            return model
        else:
            scores = cross_val_score(model, trainx, trainy, cv=cv, scoring="neg_mean_squared_error")
            scores = np.sqrt(-scores)
            print "scores", scores
            print "mean score", scores.mean()
            print "val score spend ", (time.time() - start)
            if re_fit:
                start2 = time.time()
                model.fit(trainx, trainy)
                print "fit spend ", (time.time() - start2)
                return model
            print "warning!!! no train model. if want to train, set re_fit True."
    else:
        # grid search cv

        grid_search = GridSearchCV(estimator=model,
                                   param_grid=grid_params,
                                   refit=re_fit,
                                   cv=cv,
                                   verbose=2,
                                   n_jobs=-1,
                                   scoring="neg_mean_squared_error")
        grid_search.fit(trainx, trainy)
        print "best params ", grid_search.best_params_
        scores = grid_search.best_score_
        scores = np.sqrt(-scores)
        print "grid search spend ", (time.time() - start)
        print "scores", scores
        print "mean score", scores.mean()
        model = grid_search.best_estimator_
    return model


def predict_and_save_result(model, test_x, test, save_path, transform=None):
    predicts = model.predict(test_x)
    if transform is not None:
        if transform == "log":
            predicts = np.exp(predicts) - 1
    test["trip_duration"] = predicts
    save_result(test[["id", "trip_duration"]], save_path)


def save_result(result, path):
    assert isinstance(result, pd.DataFrame)
    assert len(result.columns.values) == 2
    assert ("id" in result.columns.values and "trip_duration" in result.columns.values)
    result.to_csv(path, index=False)


def preprocess_weather(train, test):
    weather = pd.read_csv("../data/weather_data_nyc_centralpark_2016.csv")
    weather.precipitation = np.where(weather.precipitation == "T", 0.0, weather.precipitation).astype(np.float)
    weather.snow_fall = np.where(weather.snow_fall == "T", 0.0, weather.snow_fall).astype(np.float)
    weather.snow_depth = np.where(weather.snow_depth == "T", 0.0, weather.snow_depth).astype(np.float)
    weather["month"] = weather.date.map(lambda x: int(x.split("-")[1]))
    weather["day"] = weather.date.map(lambda x: int(x.split("-")[0]))
    weather.drop("date", axis=1)
    train = pd.merge(train, weather, on=["month", "day"], how="left")
    test = pd.merge(test, weather, on=["month", "day"], how="left")
    return train, test


def preprocess_time(data):
    start_time = data.pickup_datetime
    data["start_time"] = start_time.map(lambda x: pd.Timestamp(x))
    start_time = data.start_time
    data["month"] = start_time.map(lambda x: x.month)
    data["day"] = start_time.map(lambda x: x.day)
    data["hour"] = start_time.map(lambda x: x.hour)
    data["minute"] = start_time.map(lambda x: x.minute)
    data["second"] = start_time.map(lambda x: x.second)
    data["weekday"] = start_time.map(lambda x: x.weekday() + 1)
    data["dayofyear"] = start_time.map(lambda x: x.dayofyear)
    data["weekofyear"] = start_time.map(lambda x: x.weekofyear)
    data["is_weekend"] = np.where(data.weekday <= 5, 0, 1)
    return data


def preprocess_lon_lat(data):
    """
    经纬度扩大10w倍
    :param data:
    :return:
    """
    global __amplification_factor
    data.pickup_longitude = data.pickup_longitude * __amplification_factor
    data.pickup_latitude = data.pickup_latitude * __amplification_factor
    data.dropoff_longitude = data.dropoff_longitude * __amplification_factor
    data.dropoff_latitude = data.dropoff_latitude * __amplification_factor

    # 计算距离,方位
    data["haversine_distance"] = haversine_distance(data.pickup_longitude / __amplification_factor,
                                                    data.pickup_latitude / __amplification_factor,
                                                    data.dropoff_longitude / __amplification_factor,
                                                    data.dropoff_latitude / __amplification_factor)
    data["manhattan_distance"] = manhattan_distance(data.pickup_longitude / __amplification_factor,
                                                    data.pickup_latitude / __amplification_factor,
                                                    data.dropoff_longitude / __amplification_factor,
                                                    data.dropoff_latitude / __amplification_factor)
    data["bearing"] = bearing(data.pickup_longitude / __amplification_factor,
                              data.pickup_latitude / __amplification_factor,
                              data.dropoff_longitude / __amplification_factor,
                              data.dropoff_latitude / __amplification_factor)
    return data


def preprocess_step_direction(data):
    "将step direction 划分为left,right和straight的数量"
    counters = data.step_direction.map(lambda x: Counter(x.split("|")))
    data["left"] = counters.map(lambda x: x["left"])
    data["right"] = counters.map(lambda x: x["right"])
    data["straight"] = counters.map(lambda x: x["straight"])
    data["uturn"] = counters.map(lambda x: x["uturn"])
    data["slight_left"] = counters.map(lambda x: x["slight left"])
    data["slight_right"] = counters.map(lambda x: x["slight right"])
    # left,right,straight 站它们和的比例
    data["sum_all"] = data.left + data.right + data.straight + data.uturn + data.slight_left + data.slight_right
    data["n_step_other"] = data.number_of_steps - data.sum_all
    data["sum_all_is_zero"] = np.where(data.sum_all == 0, 1, 0)
    data["sum_left_right_straight"] = data.left + data.right + data.straight
    data["ratio_left_sum_all"] = (data.left / data.sum_all).fillna(0)
    data["ratio_right_sum_all"] = (data.right / data.sum_all).fillna(0)
    data["ratio_straight_sum_all"] = (data.straight / data.sum_all).fillna(0)
    data["ratio_right_left_sum_all"] = ((data.right + data.left) / data.sum_all).fillna(0)
    data["ratio_straight_right_sum_all"] = ((data.right + data.straight) / data.sum_all).fillna(0)
    data["ratio_left_straight_sum_all"] = ((data.straight + data.left) / data.sum_all).fillna(0)
    data["ratio_uturn_sum_all"] = (data.uturn / data.sum_all).fillna(0)
    data["ratio_slight_left_sum_all"] = (data.slight_left / data.sum_all).fillna(0)
    data["ratio_slight_right_sum_all"] = (data.slight_right / data.sum_all).fillna(0)
    return data


def preprocess_store_and_fwd_flag(data):
    data["store_and_fwd_int"] = np.where(data.store_and_fwd_flag == "N", 0, 1)
    return data


def preprocess_duration_and_fast_travel_time(train, test):
    duration_by_hour_vendor_id = train.groupby(["vendor_id", "hour"])[["trip_duration", "total_travel_time"]].mean()
    duration_by_hour_vendor_id.columns = ["avg_trip_duration_by_hour_vendor_id",
                                          "avg_total_travel_time_by_hour_vendor_id"]
    duration_by_hour_vendor_id["difference_by_hour_vendor_id"] = \
        duration_by_hour_vendor_id.avg_trip_duration_by_hour_vendor_id - duration_by_hour_vendor_id.avg_total_travel_time_by_hour_vendor_id
    duration_by_hour_vendor_id["ratio_by_hour_vendor_id"] = \
        duration_by_hour_vendor_id.avg_trip_duration_by_hour_vendor_id / duration_by_hour_vendor_id.avg_total_travel_time_by_hour_vendor_id
    duration_by_hour_vendor_id["ratio_actual_by_hour_vendor_id"] = \
        duration_by_hour_vendor_id.difference_by_hour_vendor_id / duration_by_hour_vendor_id.avg_trip_duration_by_hour_vendor_id
    vendor_ids = train.vendor_id.unique()
    hours = train.hour.unique()
    features = ["avg_trip_duration_by_hour_vendor_id",
                "avg_total_travel_time_by_hour_vendor_id",
                "difference_by_hour_vendor_id",
                "ratio_by_hour_vendor_id",
                "ratio_actual_by_hour_vendor_id"]
    for i in vendor_ids:
        for j in hours:
            for f in features:
                train.loc[(train.vendor_id == i) & (train.hour == j), f] = duration_by_hour_vendor_id.loc[i].loc[j][f]
                test.loc[(train.vendor_id == i) & (train.hour == j), f] = duration_by_hour_vendor_id.loc[i].loc[j][f]

    duration_by_hour = train[["hour", "trip_duration", "total_travel_time"]].groupby("hour").mean()
    duration_by_hour["hour"] = duration_by_hour.index
    duration_by_hour.columns = ["avg_trip_duration_by_hour",
                                "avg_total_travel_time_by_hour",
                                "hour", ]
    duration_by_hour["difference_by_hour"] = \
        duration_by_hour.avg_trip_duration_by_hour - duration_by_hour.avg_total_travel_time_by_hour
    duration_by_hour["ratio_by_hour"] = \
        duration_by_hour.avg_trip_duration_by_hour / duration_by_hour.avg_total_travel_time_by_hour
    duration_by_hour["ratio_actual_by_hour"] = \
        duration_by_hour.difference_by_hour / duration_by_hour.avg_trip_duration_by_hour
    train = pd.merge(train, duration_by_hour, on="hour", how="left")
    test = pd.merge(test, duration_by_hour, on="hour", how="left")

    # 高峰程度, 按照ratio_by_hour统计 界限2.1 和 2.5
    train["peek_degree"] = np.where(train.ratio_by_hour < 2.1, 0, np.where(train.ratio_by_hour < 2.5, 1, 2))
    test["peek_degree"] = np.where(test.ratio_by_hour < 2.1, 0, np.where(test.ratio_by_hour < 2.5, 1, 2))

    return train, test


def preprocess_cluster(train, test):
    lon_lim = [-74.03, -73.77]
    lat_lim = [40.6, 40.9]
    loc_df = pd.DataFrame()
    loc_df["lons"] = list(train.pickup_longitude) + list(train.dropoff_longitude)
    loc_df["lats"] = list(train.pickup_latitude) + list(train.dropoff_latitude)
    loc_df = loc_df[(loc_df.lons >= lon_lim[0]) & (loc_df.lons <= lon_lim[1])]
    loc_df = loc_df[(loc_df.lats >= lat_lim[0]) & (loc_df.lats <= lat_lim[1])]
    cluster = KMeans(n_clusters=15, random_state=2017)
    cluster.fit(loc_df)
    # label
    train["pickup_label"] = cluster.predict(train[["pickup_longitude", "pickup_latitude"]])
    train["dropoff_label"] = cluster.predict(train[["dropoff_longitude", "dropoff_latitude"]])
    test["pickup_label"] = cluster.predict(test[["pickup_longitude", "pickup_latitude"]])
    test["dropoff_label"] = cluster.predict(test[["dropoff_longitude", "dropoff_latitude"]])

    # pickup and dropoff center lon lat
    temp1 = pd.DataFrame(cluster.cluster_centers_, columns=["pickup_label_longitude", "pickup_label_latitude"])
    temp2 = pd.DataFrame(cluster.cluster_centers_, columns=["dropoff_label_longitude", "dropoff_label_latitude"])
    temp1["pickup_label"] = temp1.index
    temp2["dropoff_label"] = temp2.index
    train = pd.merge(train, temp1, on="pickup_label", how="left")
    train = pd.merge(train, temp2, on="dropoff_label", how="left")
    test = pd.merge(test, temp1, on="pickup_label", how="left")
    test = pd.merge(test, temp2, on="dropoff_label", how="left")

    # pickup_center to dropoff_center
    train["haversine_distance_pcenter_dcenter"] = haversine_distance(train.pickup_label_longitude,
                                                                     train.pickup_label_latitude,
                                                                     train.dropoff_label_longitude,
                                                                     train.dropoff_label_latitude)
    train["manhattan_distance_pcenter_dcenter"] = manhattan_distance(train.pickup_label_longitude,
                                                                     train.pickup_label_latitude,
                                                                     train.dropoff_label_longitude,
                                                                     train.dropoff_label_latitude)
    train["bearing_pcenter_dcenter"] = bearing(train.pickup_label_longitude,
                                               train.pickup_label_latitude,
                                               train.dropoff_label_longitude,
                                               train.dropoff_label_latitude)
    test["haversine_distance_pcenter_dcenter"] = haversine_distance(test.pickup_label_longitude,
                                                                    test.pickup_label_latitude,
                                                                    test.dropoff_label_longitude,
                                                                    test.dropoff_label_latitude)
    test["manhattan_distance_pcenter_dcenter"] = manhattan_distance(test.pickup_label_longitude,
                                                                    test.pickup_label_latitude,
                                                                    test.dropoff_label_longitude,
                                                                    test.dropoff_label_latitude)
    test["bearing_pcenter_dcenter"] = bearing(test.pickup_label_longitude,
                                              test.pickup_label_latitude,
                                              test.dropoff_label_longitude,
                                              test.dropoff_label_latitude)

    # pickup to pickup_center
    train["haversine_distance_p_pcenter"] = haversine_distance(train.pickup_longitude,
                                                               train.pickup_latitude,
                                                               train.pickup_label_longitude,
                                                               train.pickup_label_latitude)
    train["manhattan_distance_p_pcenter"] = manhattan_distance(train.pickup_longitude,
                                                               train.pickup_latitude,
                                                               train.pickup_label_longitude,
                                                               train.pickup_label_latitude)
    train["bearing_p_pcenter"] = bearing(train.pickup_longitude,
                                         train.pickup_latitude,
                                         train.pickup_label_longitude,
                                         train.pickup_label_latitude)
    test["haversine_distance_p_pcenter"] = haversine_distance(test.pickup_longitude,
                                                              test.pickup_latitude,
                                                              test.pickup_label_longitude,
                                                              test.pickup_label_latitude)
    test["manhattan_distance_p_pcenter"] = manhattan_distance(test.pickup_longitude,
                                                              test.pickup_latitude,
                                                              test.pickup_label_longitude,
                                                              test.pickup_label_latitude)
    test["bearing_p_pcenter"] = bearing(test.pickup_longitude,
                                        test.pickup_latitude,
                                        test.pickup_label_longitude,
                                        test.pickup_label_latitude)

    # dropoff to dropoff_center
    train["haversine_distance_d_dcenter"] = haversine_distance(train.dropoff_longitude,
                                                               train.dropoff_latitude,
                                                               train.dropoff_label_longitude,
                                                               train.dropoff_label_latitude)
    train["manhattan_distance_d_dcenter"] = manhattan_distance(train.dropoff_longitude,
                                                               train.dropoff_latitude,
                                                               train.dropoff_label_longitude,
                                                               train.dropoff_label_latitude)
    train["bearing_d_dcenter"] = bearing(train.dropoff_longitude,
                                         train.dropoff_latitude,
                                         train.dropoff_label_longitude,
                                         train.dropoff_label_latitude)
    test["haversine_distance_d_dcenter"] = haversine_distance(test.dropoff_longitude,
                                                              test.dropoff_latitude,
                                                              test.dropoff_label_longitude,
                                                              test.dropoff_label_latitude)
    test["manhattan_distance_d_dcenter"] = manhattan_distance(test.dropoff_longitude,
                                                              test.dropoff_latitude,
                                                              test.dropoff_label_longitude,
                                                              test.dropoff_label_latitude)
    test["bearing_d_dcenter"] = bearing(test.dropoff_longitude,
                                        test.dropoff_latitude,
                                        test.dropoff_label_longitude,
                                        test.dropoff_label_latitude)
    return train, test


__train = None
__test = None


def load_train():
    global __train
    if __train is None:
        print "loading train ..."
        __train = pd.read_csv("../data/train.csv")
        ft_part1 = pd.read_csv("../data/fastest_routes_train_part_1.csv")
        ft_part2 = pd.read_csv("../data/fastest_routes_train_part_2.csv")
        ft = pd.concat([ft_part1, ft_part2])
        __train = pd.merge(__train, ft, on="id")
        __train = preprocess_time(__train)
        __train = preprocess_lon_lat(__train)
        __train = preprocess_step_direction(__train)
        __train = preprocess_store_and_fwd_flag(__train)
        print "load train finish"
    return __train


def load_test():
    global __test
    if __test is None:
        print "loading test ..."
        __test = pd.read_csv("../data/test.csv")
        ft = pd.read_csv("../data/fastest_routes_test.csv")
        __test = pd.merge(__test, ft, on="id")
        __test = preprocess_time(__test)
        __test = preprocess_lon_lat(__test)
        __test = preprocess_step_direction(__test)
        __test = preprocess_store_and_fwd_flag(__test)
        print "load test finish"
    return __test


CACHE = True
CACHE_PATH = "../data/CACHE/{}_pickle"


# (1458643, 35) train
# (625134, 33) test
# cache 9s
# no cache 142s
def load_train_and_test():
    start = time.time()
    train_cache_path = CACHE_PATH.format("train")
    test_cache_path = CACHE_PATH.format("test")
    if CACHE and os.path.exists(train_cache_path) and os.path.exists(test_cache_path):
        print "read cache data"
        train = pd.read_pickle(train_cache_path)
        test = pd.read_pickle(test_cache_path)
    else:
        print "no cache read, read csv"
        train = load_train()
        test = load_test()

        # 抽取实际时间和最快时间 之间的特征
        train, test = preprocess_duration_and_fast_travel_time(train, test)
        train, test = preprocess_cluster(train, test)
        train, test = preprocess_weather(train, test)
        if CACHE:
            train.to_pickle(train_cache_path)
            test.to_pickle(test_cache_path)

    print "Time taken is ", (time.time() - start)
    print train.shape
    print test.shape

    train_null_size = pd.isnull(train).sum().sum()
    print "in train, null size is ", train_null_size
    test_null_size = pd.isnull(test).sum().sum()
    print "in test, null size is ", test_null_size

    print "train columns:", train.columns
    return train, test


def haversine_distance(lon1, lat1, lon2, lat2):  # 2点经纬度距离计算, 单位km
    """function to calculate haversine distance between two co-ordinates"""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lon1, lat2, lon2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h


def manhattan_distance(lon1, lat1, lon2, lat2):  # 2点曼哈顿距离计算, 单位km
    a = haversine_distance(lon1, lat1, lon2, lat1)
    b = haversine_distance(lon1, lat1, lon1, lat2)
    return a + b


def bearing(lng1, lat1, lng2, lat2):  # 方位计算
    """ function was taken from beluga's notebook as this function works on array
    while my function used to work on individual elements and was noticably slow"""
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))


def __main():
    train, test = load_train_and_test()
    print train


if __name__ == '__main__':
    __main()
    # print bearing(0, 180, 0, 180)
