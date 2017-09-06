#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-9-4, 14:32

@Description:

@Update Date: 17-9-4, 14:32
"""

from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans


def mini_batch_kmeans(loc_df, n_clusters, n_init=5):
    assert (("lon" in loc_df.columns[0] or "lng" in loc_df.columns[0]) and "lat" in loc_df.columns[1])
    cluster = MiniBatchKMeans(n_clusters=n_clusters, random_state=2017, n_init=n_init)
    cluster.fit(loc_df)
    return cluster


def k_means(loc_df, n_clusters, n_init=10):
    """
    :param loc_df: dataframe  第一列lons 第二列lats
    :param n_clusters:
    :return:
    """
    assert (("lon" in loc_df.columns[0] or "lng" in loc_df.columns[0]) and "lat" in loc_df.columns[1])
    cluster = KMeans(n_clusters=n_clusters, random_state=2017, n_jobs=-1, n_init=n_init)
    cluster.fit(loc_df)
    return cluster


def dbscan(loc_df, eps=1, min_samples=100, save_path=None):
    """
    :param loc_df: dataframe  第一列lons 第二列lats
    :return:
    """
    assert (("lon" in loc_df.columns[0] or "lng" in loc_df.columns[0]) and "lat" in loc_df.columns[1])
    cluster = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)
    cluster.fit(loc_df)
    if save_path is not None:
        loc_df["labels"] = cluster.labels_
        save_path += ("dbscan_eps{}_mins{}.csv").format(eps, min_samples)
        print "save in :", save_path
        loc_df.to_csv(save_path, index=False)
    return cluster
