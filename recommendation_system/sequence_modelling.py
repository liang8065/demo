#encoding:utf-8
import multiprocessing
import gensim
import sys
from random import shuffle
import cPickle as pickle

def parse_playlist_get_sequence(in_line, playlist_sequence):
    song_sequence = []
    contents = in_line.strip().split("\t")
    for song in contents[1:]:
        try:
            song_id, song_name, artist, popularity = song.split("::")
            song_sequence.append(song_id)
        except:
            print "song format error"
            print song + "\n"
    for i in range(len(song_sequence)):
        shuffle(song_sequence)
        playlist_sequence.append(song_sequence)

def train_song2vec(in_file, out_file):
    playlist_sequence = []
    for line in open(in_file):
        parse_playlist_get_sequence(line, playlist_sequence)

    cores = multiprocessing.cpu_count()
    print "using all "+str(cores)+" cores"
    print "Training word2vec model..."
    model = gensim.models.Word2Vec(sentences=playlist_sequence, size=150, min_count=3, window=7, workers=cores)
    print "Saving model..."
    model.save(out_file)

def validate():
    song_dic = pickle.load(open("../../data/recommendation_system/popular_song.pkl","rb"))
    model_str = "../../data/recommendation_system/song2vec.model"
    model = gensim.models.Word2Vec.load(model_str)

    song_id_list = song_dic.keys()[1000:1500:50]
    for song_id in song_id_list:
        result_song_list = model.most_similar(song_id)

        print song_id, song_dic[song_id]
        print "\n相似歌曲 和 相似度 分别为:"
        for song in result_song_list:
            print "\t", song_dic[song[0]], song[1]
        print "\n"

if __name__ == "__main__":
    #song_sequence_file = "../../data/recommendation_system/popular.playlist"
    #model_file = "../../data/recommendation_system/song2vec.model"
    #train_song2vec(song_sequence_file, model_file)

    validate()
