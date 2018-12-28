#encoding:utf-8
from __future__ import (absolute_import, division, print_function, unicode_literals)
import os
import io
import cPickle as pickle
from surprise import KNNBaseline, Reader
from surprise import Dataset

def load_dic():
    id_name_dic = pickle.load(open("../../data/recommendation_system/popular_playlist.pkl","rb"))
    name_id_dic = {}
    for playlist_id in id_name_dic:
        name_id_dic[id_name_dic[playlist_id]] = playlist_id

    return id_name_dic, name_id_dic


def load_train_data():
    file_path = os.path.expanduser('../../data/recommendation_system/popular_music_suprise_format.txt')
    reader = Reader(line_format='user item rating', sep=',')
    music_data = Dataset.load_from_file(file_path, reader=reader)
    trainset = music_data.build_full_trainset()

    return trainset


def compute_user_neighbors(id_name_dic, name_id_dic, trainset):
    algo = KNNBaseline()
    algo.fit(trainset)

    current_playlist = name_id_dic.keys()[1]
    print("歌单名称: ", current_playlist)
    playlist_id = name_id_dic[current_playlist]
    print("歌单id: ", playlist_id)
    playlist_inner_id = algo.trainset.to_inner_uid(playlist_id)
    print("内部id: ", playlist_inner_id)

    playlist_neighbors = algo.get_neighbors(playlist_inner_id, k=10)

    playlist_neighbors = (algo.trainset.to_raw_uid(inner_id) for inner_id in playlist_neighbors)
    playlist_neighbors = (id_name_dic[playlist_id] for playlist_id in playlist_neighbors)
    print()
    print("和歌单 《", current_playlist, "》 最接近的10个歌单为：")
    for playlist in playlist_neighbors:
        print(algo.trainset.to_inner_uid(name_id_dic[playlist]), playlist)


if __name__ == "__main__":
    id_name_dic, name_id_dic = load_dic()
    trainset = load_train_data()

    #print id_name_dic.keys()[2]
    #print(id_name_dic[id_name_dic.keys()[2]])
    #print trainset.n_items
    #print trainset.n_users

    compute_user_neighbors(id_name_dic, name_id_dic, trainset)

