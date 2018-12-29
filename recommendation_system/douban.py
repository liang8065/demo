#encoding:utf-8
from __future__ import (absolute_import, division, print_function, unicode_literals)
import os
import io
from surprise import KNNBaseline, Reader
from surprise import Dataset
from surprise.model_selection import cross_validate

def load_dic(filename):
    id_name_dic = {}
    name_id_dic = {}
    with io.open(filename, "r", encoding="utf-8") as f:
        for line in f:
            arr = line.strip().split(",")
            if len(arr) != 2:
                continue
            id_name_dic[arr[0]] = arr[1]
            name_id_dic[arr[1]] = arr[0]

    return id_name_dic, name_id_dic


def load_train_data():
    file_path = os.path.expanduser('../../data/recommendation_system/douban/suprise_format.txt')
    reader = Reader(line_format='user item rating', sep=',')
    douban_data = Dataset.load_from_file(file_path, reader=reader)
    trainset = douban_data.build_full_trainset()

    return trainset


def compute_user_neighbors(id_name_dic, name_id_dic, trainset):
    algo = KNNBaseline()
    algo.fit(trainset)

    user_name = name_id_dic.keys()[1]
    print("user_name: ", user_name)
    user_id = name_id_dic[user_name]
    print("user_id: ", user_id)
    user_inner_id = algo.trainset.to_inner_uid(user_id)
    print("内部id: ", user_inner_id)

    user_neighbors = algo.get_neighbors(user_inner_id, k=10)

    user_neighbors = (algo.trainset.to_raw_uid(inner_id) for inner_id in user_neighbors)
    user_neighbors = (id_name_dic[user_id] for user_id in user_neighbors)
    print()
    print("和user 《", user_name, "》 最接近的10个user为：")
    for user in user_neighbors:
        print(algo.trainset.to_inner_uid(name_id_dic[user]), user)


def compute_movie_neighbors(id_name_dic, name_id_dic, trainset):
    sim_options = {'user_based': False}
    algo = KNNBaseline(sim_options=sim_options)
    algo.fit(trainset)

    #movie_name = name_id_dic.keys()[1]
    movie_name = "古墓迷途2"
    print("movie_name: ", movie_name)
    movie_id = name_id_dic[movie_name]
    print("movie_id: ", movie_id)
    movie_inner_id = algo.trainset.to_inner_iid(movie_id)
    print("内部id: ", movie_inner_id)

    movie_neighbors = algo.get_neighbors(movie_inner_id, k=10)

    movie_neighbors = (algo.trainset.to_raw_iid(inner_id) for inner_id in movie_neighbors)
    movie_neighbors = (id_name_dic[movie_id] for movie_id in movie_neighbors)
    print()
    print("和movie 《", movie_name, "》 最接近的10个movie为：")
    for movie in movie_neighbors:
        print(algo.trainset.to_inner_iid(name_id_dic[movie]), movie)


def evaluate():
    file_path = os.path.expanduser('../../data/recommendation_system/douban/suprise_format.txt')
    reader = Reader(line_format='user item rating', sep=',')
    douban_data = Dataset.load_from_file(file_path, reader=reader)
    douban_data.split(n_folds=5)

    from surprise import NormalPredictor, BaselineOnly, KNNBasic, KNNWithMeans, KNNWithZScore
    from surprise import SVD, SVDpp, NMF, SlopeOne, CoClustering

    # 3.6247  3.1783
    #algo = NormalPredictor()

    # 3.3207  2.9803
    #algo = BaselineOnly()

    # 3.3693  3.0146
    #algo = KNNBasic()

    # 3.3301  2.9193
    #algo = KNNWithMeans()

    # 3.3319  2.9187
    #algo = KNNWithZScore()

    # 3.3435  2.9955
    #algo = KNNBaseline()

    # 3.3169  2.9676
    #algo = SVD()

    # 3.3173  2.9589
    #algo = SVDpp()

    # 3.4957  3.0444
    #algo = NMF()

    # 3.3515  2.9327
    #algo = SlopeOne()

    # 3.3659  2.9723
    algo = CoClustering()
    cross_validate(algo, douban_data, measures=['RMSE', 'MAE'], n_jobs=5, verbose=True)



if __name__ == "__main__":
    #id_name_dic, name_id_dic = load_dic("../../data/recommendation_system/douban/user.pkl")
    #id_name_dic, name_id_dic = load_dic("../../data/recommendation_system/douban/movie.pkl")
    #trainset = load_train_data()
    #compute_user_neighbors(id_name_dic, name_id_dic, trainset)
    #compute_movie_neighbors(id_name_dic, name_id_dic, trainset)

    evaluate()


