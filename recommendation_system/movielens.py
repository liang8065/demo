import os
import io
from surprise import KNNBaseline
from surprise import Dataset

def read_item_names():
    file_name = (os.path.expanduser('~') + '/.surprise_data/ml-100k/ml-100k/u.item')
    rid_to_name = {}
    name_to_rid = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_to_name[line[0]] = line[1]
            name_to_rid[line[1]] = line[0]

    return rid_to_name, name_to_rid


if __name__ == "__main__":
    data = Dataset.load_builtin('ml-100k')
    trainset = data.build_full_trainset()
    sim_options = {'name': 'pearson_baseline', 'user_based': False}
    algo = KNNBaseline(sim_options=sim_options)
    algo.fit(trainset)

    rid_to_name, name_to_rid = read_item_names()

    toy_story_raw_id = name_to_rid['Toy Story (1995)']
    toy_story_inner_id = algo.trainset.to_inner_iid(toy_story_raw_id)

    toy_story_neighbors = algo.get_neighbors(toy_story_inner_id, k=10)
    toy_story_neighbors = (algo.trainset.to_raw_iid(inner_id) for inner_id in toy_story_neighbors)
    toy_story_neighbors = (rid_to_name[rid] for rid in toy_story_neighbors)

    print('The 10 nearest neighbors of Toy Story are:')
    for movie in toy_story_neighbors:
        print(movie)
