from LaplaceWSAPIClient import *
from MarkowitzSerde import *
from TargetSerde import *
from Operators import *
from TargetOperators import *
from ClientConfig import client_config

config_m01 = {
    "configSelect": "gekkoConfig"
}

config_m02 = {
    "configSelect": "defaultConfig",
    "nbits": 8
}

x1 = op.sym('x1')
x2 = op.sym('x2')
expr = (op.label('l', op.minimize([x1,x2], (x1 - 0.1)**2))
                & op.label('v', x2 == 1.0)
                & op.label('v2', x1 > 0.2))
target = expressionToTarget(expr, {'l': 1.0, 'v': 100.0, 'v2': 1000.0})
query = SolverProblem(target, config_m01, UserTokenSerde(client_config["user"], client_config["token"]))

performQuerySolverProblem("ws://localhost:8799", query, lambda x: print("yahoo: ", x))

query2 = ExtractQUBO(target, config_m02, UserTokenSerde(client_config["user"], client_config["token"]))

performQueryExtraQUBO("ws://localhost:8799", query2, lambda x: print("yahoo: ", x))
