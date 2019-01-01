# encoding:utf-8

import numpy as np
import scipy as sp
import pandas as pd

def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

# load data
data_root = "../../../data/tencentCvrCompetition"
dfTrain = pd.read_csv("%s/train.csv" % data_root)
dfTest = pd.read_csv("%s/val.csv" % data_root)
dfAd = pd.read_csv("%s/ad.csv" % data_root)

# process data
dfTrain = pd.merge(dfTrain, dfAd, on="creativeID")
dfTest = pd.merge(dfTest, dfAd, on="creativeID")
y_train = dfTrain["label"].values

