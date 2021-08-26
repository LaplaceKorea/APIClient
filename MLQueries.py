from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Callable, cast
import numpy as np
import json
import pandas as pd
from UserTokenSerde import *

@dataclass
class ClassificationQuery:
    token: UserTokenSerde
    test_set: pd.DataFrame
    train_set: pd.DataFrame
    model: str
    parameters: Dict[str, Any]
    target_column: str

def prepareClassificationQuery(c: ClassificationQuery):
    train_as_string = c.train_set.to_csv()
    test_as_string = c.test_set.to_csv()
    query = {
        "user": c.token.user,
        "token": c.token.token,
        "model": c.model, 
        "parameters": c.parameters, 
        "train_set": train_as_string, 
        "test_set": test_as_string, 
        "target_column": c.target_column 
    }
    return query

@dataclass
class AutoEncoderQueryBuildup:
    window: int
    method: str
    limit: int
    n_estimators: int

@dataclass
class AutoEncoderQuery:
    token: UserTokenSerde
    model: str
    test_set: pd.DataFrame
    train_set: pd.DataFrame
    ignore: List[str]
    buildup: AutoEncoderQueryBuildup

def prepareAutoEncoderQuery(c: AutoEncoderQuery):
    train_as_string = c.train_set.to_csv()
    test_as_string = c.test_set.to_csv()
    query = {
        "user": c.token.user,
        "token": c.token.token,
        "model": c.model, 
        "train_set": train_as_string, 
        "test_set": test_as_string,     
        "ignore": c.ignore,
        "buildup": {
            "window": c.buildup.window,
            "method": c.buildup.method,
            "limit": c.buildup.limit,
            "n_estimators": c.buildup.n_estimators
        }
    }
    return query

@dataclass 
class LOBQueryBuildup:
    window: int
    method: str
    limit: int    

@dataclass 
class LOBQuery:
    token: UserTokenSerde
    model: str
    parameters: Dict[str, Any]
    test_set: pd.DataFrame
    train_set: pd.DataFrame
    ignore: List[str]
    target: str
    buildup: LOBQueryBuildup   

def prepareLOBQuery(c: LOBQuery):
    train_as_string = c.train_set.to_csv()
    test_as_string = c.test_set.to_csv()
    query = {
        "user": c.token.user,
        "token": c.token.token,
        "model": c.model, 
        "train_set": train_as_string, 
        "test_set": test_as_string,     
        "ignore": c.ignore,
        "target": c.target,
        "parameters": c.parameters,
        "buildup": {
            "window": c.buildup.window,
            "method": c.buildup.method,
            "limit": c.buildup.limit,
        }
    }
    return query    