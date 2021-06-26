from dataclasses import dataclass
from typing import Any, Dict, List, Union
import numpy as np
from UserTokenSerde import *
import orjson
import json
from dataclasses_serialization.json import JSONSerializer

@dataclass
class Markowitz:
    # min_p (w - p)^T S (w-p)
    # s.t. sum_i(p_i) = 1
    S: np.ndarray
    w: np.ndarray

@dataclass
class MarkowitzSerde:
    S: List[List[float]]
    w: List[float]

def makeMarkowitzSerde(m: Markowitz) -> MarkowitzSerde:
    s = orjson.dumps(m, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)
    return JSONSerializer.deserialize(MarkowitzSerde, json.loads(s))

def makeMarkowizt(m: MarkowitzSerde) -> Markowitz:
    return Markowitz(np.array(m.S, dtype=np.float), np.array(m.w, dtype=np.float))

@dataclass
class MarkowitzSolution:
    p: np.ndarray
    status: OperationStatusSerde

@dataclass
class MarkowitzSolutionSerde:
    p: List[float]
    status: OperationStatusSerde

def makeMarkowitzSolutionSerde(m: MarkowitzSolution) -> MarkowitzSolutionSerde:
    s = orjson.dumps(m, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)
    #print("mms:", s)
    return JSONSerializer.deserialize(MarkowitzSolutionSerde, json.loads(s))

def makeMarkowitzSolution(m: MarkowitzSolutionSerde) -> MarkowitzSolution:
    #print("solution => ", ParamSpec)
    return MarkowitzSolution(np.array(m.p, dtype=np.float), m.status)

@dataclass
class MarkowitzProblem:
    m: Markowitz
    config: Dict[str, Union[str,float,int]]
    info: UserTokenSerde

@dataclass 
class MarkowitzProblemSerde:
    m: MarkowitzSerde
    config: Dict[str, Union[str,float,int]]
    info: UserTokenSerde

def makeMarkowitzProblemSerde(m: MarkowitzProblem) -> MarkowitzProblemSerde:
    s = orjson.dumps(m, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)
    return JSONSerializer.deserialize(MarkowitzProblemSerde, json.loads(s))

def makeMarkowitzProblem(m: MarkowitzProblemSerde) -> MarkowitzProblem:
    return MarkowitzProblem(makeMarkowizt(m.m), m.config, m.info)

def serializeMarkowitzProblemQuery(m: MarkowitzProblem) -> str:
    m2 = makeMarkowitzProblemSerde(m)   
    query = '{ "__class__":"MarkowitzProblem", "query":' + orjson.dumps(m2).decode("utf-8")  + '}'
    return query

def deserializeMarkowitzSolutionResponse(r: str) -> MarkowitzSolution:
    rv = json.loads(r)
    p1 = JSONSerializer.deserialize(MarkowitzSolutionSerde, rv)
    print("p1=", p1)
    s2 = makeMarkowitzSolution(p1)
    return s2