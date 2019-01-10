#encoding=utf-8
import sys
import time
import pandas as pd
from tflearn.data_utils import pad_sequences
import random
import numpy as np
import h5py
import pickle
import argparse

PAD_ID = 0
UNK_ID = 1
NUM_VALID = 200000

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('all_data', type=str)
parser.add_argument('all_data_h5py', type=str)
parser.add_argument('id_index_pkl', type=str)
args = vars(parser.parse_args())

def create_vocabulary_dict(all_data):
    word2index = {}
    label2index = {}
    word_idx = 0

    word_list = ['PAD','UNK','CLS','SEP','unused1','unused2','unused3','unused4','unused5']
    for word in word_list:
        word2index[word] = word_idx
        word_idx += 1

    label_idx = 0
    for index, row in all_data.iterrows():
        if row["class"] not in label2index:
            label2index[row["class"]] = label_idx
            label_idx += 1

        arr = row["title"].strip().split(" ")
        for word in arr:
            if word not in word2index:
                word2index[word] = word_idx
                word_idx += 1

    return word2index, label2index


def transform_multilabel_as_multihot(label_list,label_size):
    result = np.zeros(label_size)
    result[label_list] = 1
    return result


def get_X_Y(all_data, word2index, label2index, label_size):
    X = []
    Y = []
    for index, row in all_data.iterrows():
        arr = row["title"].strip().split(" ")
        word_id_list = [word2index.get(x, UNK_ID) for x in arr]

        #arr = row["class"].strip().split("|")
        #label_list_dense = [label2index[l] for l in arr]
        #label_list_sparse = transform_multilabel_as_multihot(label_list_dense, label_size)
        label_idx = label2index[row["class"].strip()]

        X.append(word_id_list)
        Y.append(label_idx)

    max_sentence_length=20
    X = pad_sequences(X, maxlen=max_sentence_length, value=PAD_ID)

    return X, Y


def split_data(X, Y):
    xy = list(zip(X, Y))
    random.Random(10000).shuffle(xy)
    X,Y = zip(*xy)
    X = np.array(X)
    Y = np.array(Y)
    num_examples = len(X)
    num_valid = NUM_VALID
    num_train = num_examples - (num_valid + num_valid)
    train_X, train_Y = X[0 : num_train], Y[0 : num_train]
    valid_X, valid_Y = X[num_train : num_train+num_valid], Y[num_train : num_train+num_valid]
    test_X, test_Y = X[num_train+num_valid:], Y[num_train+num_valid:]

    return train_X, train_Y, valid_X, valid_Y, test_X, test_Y


def save_data(cache_file_h5py, cache_file_pickle, word2index, label2index,
        train_X, train_Y, valid_X, valid_Y, test_X, test_Y):
    f = h5py.File(cache_file_h5py, 'w')
    f['train_X'] = train_X
    f['train_Y'] = train_Y
    f['valid_X'] = valid_X
    f['valid_Y'] = valid_Y
    f['test_X'] = test_X
    f['test_Y'] = test_Y
    f.close()

    with open(cache_file_pickle, 'ab') as target_file:
        pickle.dump((word2index,label2index), target_file)


print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + ": begin preprocess..."); sys.stdout.flush()

# 读数据
all_data = pd.read_csv(args['all_data'], names=["class", "title"], sep='\t', encoding="utf-8")
all_data = all_data.fillna('')
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + ": finish reading data..."); sys.stdout.flush()

# 生成词典
word2index, label2index = create_vocabulary_dict(all_data)
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + ": finish creating vocabulary dict..."); sys.stdout.flush()

# 训练数据id化
label_size = len(label2index)
X, Y = get_X_Y(all_data, word2index, label2index, label_size)
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + ": finish changing data to ids..."); sys.stdout.flush()

# 训练数据分割
train_X, train_Y, valid_X, valid_Y, test_X, test_Y = split_data(X, Y)
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + ": finish split data..."); sys.stdout.flush()
print train_X.shape
print train_Y.shape
print valid_X.shape
print valid_Y.shape
print test_X.shape
print test_Y.shape

# 序列化数据和词典
save_data(args["all_data_h5py"], args["id_index_pkl"], word2index, label2index,
        train_X, train_Y, valid_X, valid_Y, test_X, test_Y)


