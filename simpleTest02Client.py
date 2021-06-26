from LaplaceWSAPIClient import *
from MarkowitzSerde import *
from ClientConfig import client_config

config_m01 = {
    "configSelect": "gekkoConfig"
}

query = MarkowitzProblem(Markowitz(np.array([[1.0,0.2,0.1],[0.2,1.0,0.3],[0.1,0.3,1.0]], dtype=np.float), np.array([0.05,0.05,0.05], dtype=np.float)), 
                config_m01, UserTokenSerde(client_config["user"], client_config["token"]))

performQuery("ws://localhost:8799", query, lambda x: print("yahoo: ", x)) #..65

d = ""
with open("query-classification.json", "r") as io:
    d = io.readline()
data = json.loads(d)
data["user"] = client_config["user"]
data["token"] = client_config["token"]

perforQueryAny("ws://localhost:8799", data, "MLWrapper", lambda x: print("hoho", x))

d = ""
with open("query-autoencoder.json", "r") as io:
    d = io.readline()
data = json.loads(d)
data["user"] = client_config["user"]
data["token"] = client_config["token"]

perforQueryAny("ws://localhost:8799", data, "MLWrapper", lambda x: print("hoho", x))

d = ""
with open("query-lob.json", "r") as io:
    d = io.readline()
data = json.loads(d)
data["user"] = client_config["user"]
data["token"] = client_config["token"]

perforQueryAny("ws://localhost:8799", data, "MLWrapper", lambda x: print("hoho", x))

d = ""
with open("query-lob-cpu.json", "r") as io:
    d = io.readline()
data = json.loads(d)
data["user"] = client_config["user"]
data["token"] = client_config["token"]

perforQueryAny("ws://localhost:8799", data, "MLWrapper", lambda x: print("hoho", x))


