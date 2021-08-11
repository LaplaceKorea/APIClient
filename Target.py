from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Callable, Union
from UserTokenSerde import *
import numpy as np

@dataclass
class QUBO:
    W: np.ndarray

@dataclass 
class Ising:
    J: np.ndarray
    v: np.ndarray 
    c: float

def QUBOToIsing(qubo: QUBO) -> Ising:
    J = np.zeros(qubo.W.shape, dtype=np.float)
    v = np.zeros((qubo.W.shape[0],), dtype=np.float)
    c = 0.0
    for i in range(qubo.W.shape[0]):
        for j in range(qubo.W.shape[1]):
            # I say x \in {0,1} = 1/2(y+1)
            J[i,j] = J[i,j] + 0.25*qubo.W[i,j]
            v[i] = v[i] + 0.25*qubo.W[i,j]
            v[j] = v[j] + 0.25*qubo.W[i,j]
            c = c + 0.25*qubo.W[i,j]
    return Ising(J,v,c)

def IsingToQUBO(ising: Ising) -> QUBO:
    W = np.zeros(ising.J.shape, dtype=np.float)
    for i in range(ising.J.shape[0]):
        for j in range(ising.J.shape[1]):
            W[i,j] = 4.0*ising.J[i,j]
    return QUBO(W)

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
    ForceNBits: List[int]
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
