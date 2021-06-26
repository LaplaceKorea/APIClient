from LaplaceWSAPIClient import *
from MarkowitzSerde import *
from ClientConfig import client_config

config_m01 = {
    "configSelect": "gekkoConfig"
}

query = MarkowitzProblem(Markowitz(np.array([[1.0,0.2,0.1],[0.2,1.0,0.3],[0.1,0.3,1.0]], dtype=np.float), np.array([0.05,0.05,0.05], dtype=np.float)), 
                config_m01, UserTokenSerde(client_config["user"], client_config["token"]))

performQuery(client_config["wss"], query, lambda x: print("yahoo: ", x)) 
