# use env: amazon-braket
from dataclasses import dataclass
from datetime import datetime
import time
from typing import Any, Dict, List, Tuple, Callable, Union
from typing_extensions import ParamSpecKwargs
from UserTokenSerde import *
import numpy as np
from dataclasses_serialization.json import JSONSerializer
import orjson
import json
import os

@dataclass
class Trade:
    Step: int
    Buy: bool
    Ticker: str
    Qty: float
    Price: float

@dataclass
class SimulationStep:
    Step: int
    Pnl: float
    Portfolio: Dict[str,float]
    Trades: List[Trade]

@dataclass
class RLQuery:
    EnvId: str # "default"
    From: datetime
    To: datetime
    InitialPortfolio: Dict[str,float] # special key: BankAccount
    Info: UserTokenSerde 

@dataclass 
class RLQuerySerde:
    EnvId: str # "default"
    From: str
    To: str
    InitialPortfolio: Dict[str,float] # special key: BankAccount
    Info: UserTokenSerde 

def makeRLQuery(q: RLQuerySerde) -> RLQuery:
    return RLQuery(
        q.EnvId,
        datetime(*time.strptime(q.From, "%Y-%m-%dT%H:%M:%S")[:6]),
        datetime(*time.strptime(q.To, "%Y-%m-%dT%H:%M:%S")[:6]),
        q.InitialPortfolio,
        q.Info
    )

def makeRLQuerySerde(q: RLQuery) -> RLQuerySerde:
    return RLQuerySerde (
        q.EnvId,
        q.From.strftime("%Y-%m-%dT%H:%M:%S"),
        q.To.strftime("%Y-%m-%dT%H:%M:%S"),
        q.InitialPortfolio,
        q.Info
    )

@dataclass
class RLResult:
    Steps: List[List[SimulationStep]]
    Status: OperationStatusSerde

def serializeRLQueryQuery(q: RLQuery) -> str:
    q2 = makeRLQuerySerde(q)
    query = '{ "__class__":"RLQuery", "query":' + orjson.dumps(q2).decode("utf-8") + '}'
    return query
#def serializeExtractQUBOQuery(q: ExtractQUBO) -> str:
#    q2 = makeExtractQUBOSerde(q)
#    query = '{ "__class__":"ExtractQUBO", "query":' + orjson.dumps(q2).decode("utf-8") + '}'
#    #print(query)
#    return query   
