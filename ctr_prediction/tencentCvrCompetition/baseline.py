# encoding:utf-8

import zipfile
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
y_test = dfTest["label"].values

# model building
key = "appID"
dfCvr = dfTrain.groupby(key).apply(lambda df: np.mean(df["label"])).reset_index()
dfCvr.columns = [key, "avg_cvr"]
dfTest = pd.merge(dfTest, dfCvr, how="left", on=key)
dfTest["avg_cvr"].fillna(np.mean(dfTrain["label"]), inplace=True)
proba_test = dfTest["avg_cvr"].values

loss = logloss(y_test, proba_test)
print loss
