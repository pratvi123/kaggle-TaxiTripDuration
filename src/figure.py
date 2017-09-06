#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-9-1, 10:21

@Description:

@Update Date: 17-9-1, 10:21
"""

from util import *
from cluster import dbscan
import seaborn as sns
sns.despine()
def main():
    train, test = load_train_and_test()
    xlim = [-74.03, -73.77]
    ylim = [40.6, 40.9]

    train_copy = train[["pickup_longitude", "dropoff_longitude", "pickup_latitude", "dropoff_latitude"]].copy()

    train_copy = train_copy[(train_copy.pickup_longitude >= xlim[0]) & (train_copy.pickup_longitude <= xlim[1])]
    train_copy = train_copy[(train_copy.dropoff_longitude >= xlim[0]) & (train_copy.dropoff_longitude <= xlim[1])]
    train_copy = train_copy[(train_copy.pickup_latitude >= ylim[0]) & (train_copy.pickup_latitude <= ylim[1])]
    train_copy = train_copy[(train_copy.dropoff_latitude >= ylim[0]) & (train_copy.dropoff_latitude <= ylim[1])]

    test_copy = test[["pickup_longitude", "dropoff_longitude", "pickup_latitude", "dropoff_latitude"]].copy()

    test_copy = test_copy[(test_copy.pickup_longitude >= xlim[0]) & (test_copy.pickup_longitude <= xlim[1])]
    test_copy = test_copy[(test_copy.dropoff_longitude >= xlim[0]) & (test_copy.dropoff_longitude <= xlim[1])]
    test_copy = test_copy[(test_copy.pickup_latitude >= ylim[0]) & (test_copy.pickup_latitude <= ylim[1])]
    test_copy = test_copy[(test_copy.dropoff_latitude >= ylim[0]) & (test_copy.dropoff_latitude <= ylim[1])]

    train_lons = list(train_copy.pickup_longitude) + list(train_copy.dropoff_longitude)
    train_lats = list(train_copy.pickup_latitude) + list(train_copy.dropoff_latitude)

    test_lons = list(test_copy.pickup_longitude) + list(test_copy.dropoff_longitude)
    test_lats = list(test_copy.pickup_latitude) + list(test_copy.dropoff_latitude)

    loc_df = pd.DataFrame()
    loc_df["lons"] = train_lons * (10 ** 3)
    loc_df["lats"] = train_lats * (10 ** 3)

    loc_df = loc_df.sample(50000)
    dbscan(loc_df, eps=1, min_samples=100, save_path="../data/CLUSTER/")


if __name__ == '__main__':
    main()
