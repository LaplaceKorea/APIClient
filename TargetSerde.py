from dataclasses import dataclass
from typing import Any, Dict, List, Union, cast
import numpy as np
from UserTokenSerde import *
import orjson
import json
from dataclasses_serialization.json import JSONSerializer
from Target import *

@dataclass
class QUBOSerde:
    W: List[List[float]]

def makeQUBOSerde(m: QUBO) -> QUBOSerde:
    s = orjson.dumps(m, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)
    ## print(m, s)
    return JSONSerializer.deserialize(QUBOSerde, json.loads(s))

def makeQUBO(m: QUBOSerde) -> QUBO:
    return QUBO(np.array(m.W, dtype=np.float))

# not really serializable
#"""
#@dataclass
#class EmbeddedQUBO:
#    Qubo: QUBO
#    Interpretation: Dict[str, Callable[[np.ndarray],np.ndarray]]
#"""

@dataclass
class EmbeddingSerde:
    Name: str
    W: List[List[float]]

def makeEmbeddingSerde(e: Embedding) -> EmbeddingSerde:
    s = orjson.dumps(e, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)
    return JSONSerializer.deserialize(EmbeddingSerde, json.loads(s))    

def makeEmbedding(e: EmbeddingSerde) -> Embedding:
    return Embedding(e.Name, np.array(e.W, dtype=np.float))

@dataclass 
class TermSerde:
    pass

@dataclass
class MinSerde(Term):
    Coef: str
    Embed: EmbeddingSerde
    Quadratic: List[List[float]]
    Linear: Union[List[float],List[List[float]]]

def makeMinSerde(m: Min) -> MinSerde:
    s = orjson.dumps(m, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)
    return JSONSerializer.deserialize(MinSerde, json.loads(s))    

def makeMin(m: MinSerde) -> Min:
    return Min(m.Coef,makeEmbedding(m.Embed), 
                    np.array(m.Quadratic, dtype=np.float),
                    np.array(m.Linear, dtype=np.float))

@dataclass 
class MaxSerde(TermSerde):
    Coef: str
    Embed: EmbeddingSerde
    Quadratic: List[List[float]]
    Linear: Union[List[float],List[List[float]]]

def makeMaxSerde(m: Max) -> MaxSerde:
    s = orjson.dumps(m, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)
    return JSONSerializer.deserialize(MaxSerde, json.loads(s))    

def makeMax(m: MaxSerde) -> Max:
    return Max(m.Coef,makeEmbedding(m.Embed), 
                    np.array(m.Quadratic, dtype=np.float),
                    np.array(m.Linear, dtype=np.float))

@dataclass 
class EqualSerde(TermSerde):
    Coef: str
    Embed: EmbeddingSerde
    Linear: Union[List[float],List[List[float]]]
    Comparand: Union[List[List[float]],List[float]]

def makeEqualSerde(e: Equal) -> EqualSerde:
    s = orjson.dumps(e, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)
    return JSONSerializer.deserialize(EqualSerde, json.loads(s))    

def makeEqual(m: EqualSerde) -> Equal:
    return Equal(m.Coef,makeEmbedding(m.Embed), 
                    np.array(m.Linear, dtype=np.float),
                    np.array(m.Comparand, dtype=np.float))    

@dataclass
class LessThanSerde(TermSerde):
    Coef: str
    Embed: EmbeddingSerde
    Linear: Union[List[float],List[List[float]]]   
    Comparand: Union[List[List[float]],List[float]]

def makeLessThanSerde(e: LessThan) -> LessThanSerde:
    s = orjson.dumps(e, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)
    return JSONSerializer.deserialize(LessThanSerde, json.loads(s))    

def makeLessThan(m: LessThanSerde) -> LessThan:
    return LessThan(m.Coef,makeEmbedding(m.Embed), 
                    np.array(m.Linear, dtype=np.float),
                    np.array(m.Comparand, dtype=np.float))    

@dataclass
class GreaterThanSerde(TermSerde):
    Coef: str
    Embed: EmbeddingSerde
    Linear: Union[List[float],List[List[float]]]
    Comparand: Union[List[List[float]],List[float]]

def makeGreaterThanSerde(e: GreaterThan) -> GreaterThanSerde:
    s = orjson.dumps(e, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)
    return JSONSerializer.deserialize(GreaterThanSerde, json.loads(s))    

def makeGreaterThan(m: GreaterThanSerde) -> GreaterThan:
    return GreaterThan(m.Coef,makeEmbedding(m.Embed), 
                    np.array(m.Linear, dtype=np.float),
                    np.array(m.Comparand, dtype=np.float))    

@dataclass
class TermSerdeNode(TermSerde):
    maxTerm: List[MaxSerde]
    minTerm: List[MinSerde]
    equalTerm: List[EqualSerde]
    lessThanTerm: List[LessThanSerde]
    greaterThanTerm: List[GreaterThanSerde]

def makeTermSerde(t: Term)->TermSerde:
    maxTerm: List[MaxSerde] = []
    minTerm: List[MinSerde] = []
    equalTerm: List[EqualSerde] = []
    lessThanTerm: List[LessThanSerde] = []
    greaterThanTerm: List[GreaterThanSerde] = []

    if isinstance(t, Max):
        maxTerm = [makeMaxSerde(cast(Max,t))]
    if isinstance(t, Min):
        minTerm = [makeMinSerde(cast(Min,t))]
    if isinstance(t,Equal):
        equalTerm = [makeEqualSerde(cast(Equal,t))]
    if isinstance(t,LessThan):
        lessThanTerm = [makeLessThanSerde(cast(LessThan,t))]
    if isinstance(t,GreaterThan):
        greaterThanTerm = [makeGreaterThanSerde(cast(GreaterThan,t))]
    return TermSerdeNode(maxTerm, minTerm, equalTerm, lessThanTerm, greaterThanTerm)

def makeTerm(t: TermSerde)->Term:
    #print("makeTerm>", t)
    assert(isinstance(t, TermSerdeNode))
    tsn = cast(TermSerdeNode,t)
    if len(tsn.maxTerm) > 0:
        return makeMax(tsn.maxTerm[0])
    if len(tsn.minTerm) > 0:
        return makeMin(tsn.minTerm[0])
    if len(tsn.equalTerm) > 0:
        return makeEqual(tsn.equalTerm[0])
    if len(tsn.lessThanTerm) > 0:
        return makeLessThan(tsn.lessThanTerm[0])
    if len(tsn.greaterThanTerm) > 0:
        return makeGreaterThan(tsn.greaterThanTerm[0])
    raise Exception("structure is void")

# Interpretation is not easily translatable and thus removed!
#@dataclass
#class Target:
#    Terms: List[Term]
#    Parameters: Dict[str, float]
#    Initialization: Dict[str, Tuple[Embedding, np.ndarray]]
#    Interpretation: Dict[str, Callable[[np.ndarray],np.ndarray]]    

@dataclass
class TargetSerde:
    Terms: List[TermSerdeNode]
    ForceNBits: List[int]
    Parameters: Dict[str, float]
    Initialization: Dict[str, Tuple[EmbeddingSerde, Union[List[float],List[List[float]]]]]

def makeTargetSerde(t: Target) -> TargetSerde:
    #t2 = Target(t.Terms, t.Parameters, t.Initialization, {}) ~ no/no/no
    #s = orjson.dumps(t2, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)
    #return JSONSerializer.deserialize(TargetSerde, json.loads(s)) 
    init: Dict[str, Tuple[EmbeddingSerde, Union[List[float],List[List[float]]] ]] = {}
    for k in t.Initialization:
        embed, value = t.Initialization[k]
        init[k] = (makeEmbeddingSerde(embed), value.tolist())
    return TargetSerde([cast(TermSerdeNode, makeTermSerde(s)) for s in t.Terms], t.ForceNBits, t.Parameters, init)

def makeTarget(t: TargetSerde) -> Target:
    init: Dict[str, Tuple[Embedding,np.ndarray]] = {}
    for k in t.Initialization:
        embed, value = t.Initialization[k]
        init[k] = (makeEmbedding(embed), np.ndarray(value, dtype=np.float))
    return Target([makeTerm(s) for s in t.Terms], t.ForceNBits, t.Parameters, init, {})

# not for now
#"""
#@dataclass
#class EigenSystem:
#    W: np.ndarray
#"""

@dataclass
class SolverProblemSerde:
    Description: TargetSerde
    Config: Dict[str, Union[str,float,int]]
    Info: UserTokenSerde

def makeSolverProblemSerde(p: SolverProblem) -> SolverProblemSerde:
    # orjson.dumps(p, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)
    # return JSONSerializer.deserialize(SolverProblemSerde, json.loads(s))  
    return SolverProblemSerde(makeTargetSerde(p.Description), p.Config, p.Info)

def makeSolverProblem(p: SolverProblemSerde) -> SolverProblem:
    sp = SolverProblem(makeTarget(p.Description), p.Config, p.Info)
    sp.Description.Interpretation['p'] = lambda x:x
    return sp

# solve:  -> Dict[str,np.ndarray]:
@dataclass
class SolverSolutionSerde:
    Results: Dict[str, Union[List[float],List[List[float]]]]
    Status: OperationStatusSerde

def makeSolverSolutionSerde(s: SolverSolution) -> SolverSolutionSerde:
    s = orjson.dumps(s, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)
    return JSONSerializer.deserialize(SolverSolutionSerde, json.loads(s))     

def makeSolverSolution(s: SolverSolutionSerde) -> SolverSolution:
    res: Dict[str,np.ndarray] = {}
    for k in s.Results:
        res[k] = np.array(s.Results[k], dtype=float)
    return SolverSolution(res, s.Status)

@dataclass
class ExtractQUBOSerde:
    Description: TargetSerde
    Config: Dict[str, Union[str,float,int]]
    Info: UserTokenSerde

def makeExtractQUBOSerde(e: ExtractQUBO) -> ExtractQUBO:
    #s = orjson.dumps(e, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)
    #return JSONSerializer.deserialize(SolverSolutionSerde, json.loads(s))   
    return ExtractQUBOSerde(makeTargetSerde(e.Description), e.Config, e.Info)

def makeExtractQUBO(e: ExtractQUBOSerde) -> ExtractQUBO:
    return ExtractQUBO(makeTarget(e.Description), e.Config, e.Info)

@dataclass
class QUBOSolutionSerde:
    Qubo: QUBOSerde
    Status: OperationStatusSerde

def makeQUBOSolutionSerde(q: QUBOSolution) -> QUBOSolutionSerde:
    s = orjson.dumps(q, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)
    return JSONSerializer.deserialize(QUBOSolutionSerde, json.loads(s))   

def makeQUBOSolution(q: QUBOSolutionSerde) -> QUBOSolution:
    return QUBOSolution(makeQUBO(q.Qubo), q.Status)

def serializeSolverProblemQuery(p: SolverProblem) -> str:
    p2 = makeSolverProblemSerde(p)
    query = '{ "__class__":"SolverProblem", "query":' + orjson.dumps(p2).decode("utf-8") + '}'
    return query

def serializeExtractQUBOQuery(q: ExtractQUBO) -> str:
    q2 = makeExtractQUBOSerde(q)
    query = '{ "__class__":"ExtractQUBO", "query":' + orjson.dumps(q2).decode("utf-8") + '}'
    #print(query)
    return query    

def deserializeSolverSolutionResponse(r: str) -> SolverSolution:
    rv = json.loads(r)
    p1 = JSONSerializer.deserialize(SolverSolutionSerde, rv)
    #print(p1)
    p2 = makeSolverSolution(p1)
    return p2

def deserializeQUBOSolutionResponse(r: str) -> QUBOSolution:
    rv = json.loads(r)
    p1 = JSONSerializer.deserialize(QUBOSolutionSerde, rv)
    p2 = makeQUBOSolution(p1)
    return p2    
