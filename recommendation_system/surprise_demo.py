## https://surprise.readthedocs.io/en/stable/getting_started.html

## Automatic cross-validation
def func1():
    from surprise import SVD
    from surprise import Dataset
    from surprise.model_selection import cross_validate

    data = Dataset.load_builtin('ml-100k')
    algo = SVD()
    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


## Train-test split and the fit() method
def func2():
    from surprise import SVD
    from surprise import Dataset
    from surprise import accuracy
    from surprise.model_selection import train_test_split

    data = Dataset.load_builtin('ml-100k')
    trainset, testset = train_test_split(data, test_size=.25)
    algo = SVD()
    algo.fit(trainset)
    predictions = algo.test(testset)
    accuracy.rmse(predictions)


## Train on a whole trainset and the predict() method
def func3():
    from surprise import KNNBasic
    from surprise import Dataset

    data = Dataset.load_builtin('ml-100k')
    trainset = data.build_full_trainset()
    algo = KNNBasic()
    algo.fit(trainset)
    uid = str(196)  # raw user id (as in the ratings file). They are **strings**!
    iid = str(302)  # raw item id (as in the ratings file). They are **strings**!
    pred = algo.predict(uid, iid, r_ui=4, verbose=True)


## Use a custom dataset
# To load a dataset from a file
def func4():
    import os
    from surprise import BaselineOnly
    from surprise import Dataset
    from surprise import Reader
    from surprise.model_selection import cross_validate

    file_path = os.path.expanduser('~/.surprise_data/ml-100k/ml-100k/u.data')
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file(file_path, reader=reader)
    cross_validate(BaselineOnly(), data, verbose=True)


# To load a dataset from a pandas dataframe
def func5():
    import pandas as pd
    from surprise import NormalPredictor
    from surprise import Dataset
    from surprise import Reader
    from surprise.model_selection import cross_validate

    ratings_dict = {'itemID': [1, 1, 1, 2, 2],
                    'userID': [9, 32, 2, 45, 'user_foo'],
                    'rating': [3, 2, 4, 3, 1]}
    df = pd.DataFrame(ratings_dict)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
    cross_validate(NormalPredictor(), data, cv=2, verbose=True)


# Use cross-validation iterators
def func6():
    from surprise import SVD
    from surprise import Dataset
    from surprise import accuracy
    from surprise.model_selection import KFold

    data = Dataset.load_builtin('ml-100k')
    kf = KFold(n_splits=3)
    algo = SVD()
    for trainset, testset in kf.split(data):
        algo.fit(trainset)
        predictions = algo.test(testset)
        accuracy.rmse(predictions, verbose=True)


# PredefinedKFold
def func7():
    import os
    from surprise import SVD
    from surprise import Dataset
    from surprise import Reader
    from surprise import accuracy
    from surprise.model_selection import PredefinedKFold

    files_dir = os.path.expanduser('~/.surprise_data/ml-100k/ml-100k/')
    reader = Reader('ml-100k')

    train_file = files_dir + 'u%d.base'
    test_file = files_dir + 'u%d.test'
    folds_files = [(train_file % i, test_file % i) for i in (1, 2, 3, 4, 5)]

    data = Dataset.load_from_folds(folds_files, reader=reader)
    pkf = PredefinedKFold()

    algo = SVD()
    for trainset, testset in pkf.split(data):
        algo.fit(trainset)
        predictions = algo.test(testset)
        accuracy.rmse(predictions, verbose=True)


# Tune algorithm parameters with GridSearchCV
def func8():
    from surprise import SVD
    from surprise import Dataset
    from surprise.model_selection import GridSearchCV

    data = Dataset.load_builtin('ml-100k')
    param_grid = {'n_epochs': [5, 10],
                  'lr_all': [0.002, 0.005],
                  'reg_all': [0.4, 0.6]}
    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)

    gs.fit(data)

    print(gs.best_score['rmse'])
    print(gs.best_params['rmse'])


func8()
