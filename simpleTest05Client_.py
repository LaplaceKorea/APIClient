from LaplaceWSAPIClient import *
from MLQueries import *
import pandas as pd
from io import StringIO
from ClientConfig import client_config

ut = UserTokenSerde(client_config["user"],client_config["token"])
model ="xgboost_python"
target_column = "variety"
test = pd.read_csv('iris-test.csv')
train = pd.read_csv('iris-train.csv')
uri = client_config["wss"]

def listIt(x):
    for n in x:
        print(n)
        if n == "result":
            res = pd.read_csv(StringIO(x[n]))
            print(res.info())


query = ClassificationQuery(ut, test, train, model, {}, target_column)
perforQueryAny(uri, prepareClassificationQuery(query), "MLWrapper", lambda x: listIt(x))
