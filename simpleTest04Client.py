from LaplaceWSAPIClient import *
from MarkowitzSerde import *
from TargetSerde import *
from Operators import *
from TargetOperators import *
from RLStructure import *
from ClientConfig import client_config

query = RLQuery("default", datetime(2021,1,1), datetime(2021,1,21), {
            "BankAccount": 100000.0, 
            "MMM":1.0,
            "AA":1.0,
            "AXP":1.0,
            "BA":1.0,
            "BAC":1.0,
            "C":1.0,
            "CAT":1.0,
            "CVX":1.0,
            "DD":1.0,
            "DIS":1.0,
            "GE":1.0,
            "GM":1.0,
            "HD":1.0,
            "HPQ":1.0,
            "IBM":1.0,
            "JNJ":1.0,
            "JPM":1.0,
            "KO":1.0,
            "MCD":1.0,
            "MRK":1.0,
            "PFE":1.0,
            "PG":1.0,
            "T":1.0,
            "UTX":1.0,
            "VZ":1.0,
            "WMT":1.0,
            "XOM":1.0
}, UserTokenSerde(client_config["user"],client_config["token"]))

performQueryRLQuery("ws://localhost:8799", query, lambda x: print("yahoo: ", x.Steps[0][0], x.Steps[0][1], x.Steps[0][203]))
