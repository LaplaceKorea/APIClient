from LaplaceWSAPIClient import *
from MLQueries import *
import pandas as pd
from io import StringIO
from ClientConfig import client_config

ut = UserTokenSerde(client_config["user"],client_config["token"])
target_column = "variety"
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

n = test["bestr"].values.shape[0]
print(n)
lastdiff = 0.0
for i in range(n):
    if i+225 < n:
        lastdiff = test["bestr"].values[i+225] - test["bestr"].values[i]
        test["bestr"].values[i] = lastdiff
    else:
        test["bestr"].values[i] = lastdiff

n = train["bestr"].values.shape[0]
print(n)
lastdiff = 0.0
for i in range(n):
    if i+225 < n:
        lastdiff = train["bestr"].values[i+225] - train["bestr"].values[i]
        train["bestr"].values[i] = lastdiff
    else:
        train["bestr"].values[i] = lastdiff

uri = client_config["wss"]

def listIt(x):
    for n in x:
        print(n)
        if n == "result":
            res = pd.read_csv(StringIO(x[n]))
            print(res.info())

default_parameters = {
    "feat_d": 1350,
    "hidden_d": 16,
    "boost_rate": 1.0,
    "lr": 0.005,
    "L2": 0.001, # ? .0e-3
    "num_nets": 40,
    "batch_size": 2048,
    "epochs_per_stage": 1,
    "correct_epoch": 1,
    "model_order": "second",
    "normalization": True,
    "cv": True,
    "cuda": True,
    "out_f": "xxx.pth"
}

#{"model": "grownet_python", "parameters": {"feat_d": 1350, "hidden_d": 16, "boost_rate": 1.0, "lr": 0.005, "L2": 0.001, "num_nets": 40,
#  "batch_size": 2048, "epochs_per_stage": 1, "correct_epoch": 1, "model_order": "second", "normalization": true, "cv": true, "cuda": true, "out_f": "xxx.pth"},

buildup = LOBQueryBuildup(30, "append", 100)
query = LOBQuery(ut, "grownet_python", default_parameters, test, train, ["T"], "bestr", buildup)
perforQueryAny(uri, prepareLOBQuery(query), "MLWrapper", lambda x: listIt(x))
