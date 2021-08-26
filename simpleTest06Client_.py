from LaplaceWSAPIClient import *
from MLQueries import *
import pandas as pd
from io import StringIO
from ClientConfig import client_config

ut = UserTokenSerde(client_config["user"],client_config["token"])
target_column = "variety"
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')
uri = client_config["wss"]

def listIt(x):
    for n in x:
        print(n)
        if n == "result":
            res = pd.read_csv(StringIO(x[n]))
            print(res.info())


buildup = AutoEncoderQueryBuildup(30, "append", 100, 200)
query = AutoEncoderQuery(ut, "eforest_python", test, train, ["T"], buildup)
perforQueryAny(uri, prepareAutoEncoderQuery(query), "MLWrapper", lambda x: listIt(x))
