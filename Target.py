from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Callable, Union
from UserTokenSerde import *
import numpy as np

@dataclass
class QUBO:
    W: np.ndarray

@dataclass
class EmbeddedQUBO:
    Qubo: QUBO
    Interpretation: Dict[str, Callable[[np.ndarray],np.ndarray]]

@dataclass
class Embedding:
    Name: str
    W: np.ndarray

@dataclass
class Term:
    pass

@dataclass
class Min(Term):
    Coef: str
    Embed: Embedding
    Quadratic: np.ndarray
    Linear: np.ndarray

@dataclass 
class Max(Term):
    Coef: str
    Embed: Embedding
    Quadratic: np.ndarray
    Linear: np.ndarray

@dataclass 
class Equal(Term):
    Coef: str
    Embed: Embedding
    Linear: np.ndarray
    Comparand: np.ndarray 

@dataclass
class LessThan(Term):
    Coef: str
    Embed: Embedding
    Linear: np.ndarray    
    Comparand: np.ndarray

@dataclass
class GreaterThan(Term):
    Coef: str
    Embed: Embedding
    Linear: np.ndarray
    Comparand: np.ndarray

@dataclass
class Target:
    Terms: List[Term]
    Parameters: Dict[str, float]
    Initialization: Dict[str, Tuple[Embedding, np.ndarray]]
    Interpretation: Dict[str, Callable[[np.ndarray],np.ndarray]]    

@dataclass
class EigenSystem:
    W: np.ndarray

@dataclass
class SolverProblem:
    Description: Target
    Config: Dict[str, Union[str,float,int]]
    Info: UserTokenSerde

# solve:  -> Dict[str,np.ndarray]:
@dataclass
class SolverSolution:
    Results: Dict[str, np.ndarray]
    Status: OperationStatusSerde

@dataclass
class ExtractQUBO:
    Description: Target
    Config: Dict[str, Union[str,float,int]]
    Info: UserTokenSerde

@dataclass
class QUBOSolution:
    Qubo: QUBO
    Status: OperationStatusSerde
