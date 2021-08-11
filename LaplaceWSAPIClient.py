import asyncio
from dataclasses_serialization.serializer_base.serializer import Serializer
import websockets
import json
from MarkowitzSerde import *
from TargetSerde import *
from RLStructure import *

async def query(uri: str, m: MarkowitzProblem, onCompletion: Callable[[MarkowitzSolution],None]):
    #uri = "ws://localhost:8765"
    async with websockets.connect(uri, max_size=2**25, ping_timeout=None) as websocket:
        await websocket.send(serializeMarkowitzProblemQuery(m))
        rv = await websocket.recv()
        r = deserializeMarkowitzSolutionResponse(rv)
        print("result=", r)
        onCompletion(r)

async def queryAny(uri: str, q:Dict[str, Any], n:str, onCompletion: Callable[[Dict[str,Any]],None]):
    #uri = "ws://localhost:8765"
    async with websockets.connect(uri, max_size=2**25, ping_timeout=None) as websocket:
        query = '{ "__class__":\"' + n + '\", "query":' + json.dumps(q)  + '}'
        await websocket.send(query)
        async for message in websocket:
            r = json.loads(message)
            onCompletion(r)
            break
        #rv = await websocket.recv()
        #r = json.loads(rv)
        #print("result=", r)
        #onCompletion(r)

async def querySolverProblem(uri: str, m: SolverProblem, onCompletion: Callable[[SolverSolution],None]):
    async with websockets.connect(uri, max_size=2**25, ping_timeout=None) as websocket:
        await websocket.send(serializeSolverProblemQuery(m))
        rv = await websocket.recv()
        r = deserializeSolverSolutionResponse(rv)
        v = r.Results['p']
        for k in m.Description.Interpretation:
            r.Results[k] = m.Description.Interpretation[k](v)
        print("result=", r)
        onCompletion(r)   

async def queryExtraQUBO(uri: str, m: ExtractQUBO, onCompletion: Callable[[QUBOSolution],None]):
    async with websockets.connect(uri, max_size=2**25, ping_timeout=None) as websocket:
        await websocket.send(serializeExtractQUBOQuery(m))
        rv = await websocket.recv()
        r = deserializeQUBOSolutionResponse(rv)
        print("result=", r)
        onCompletion(r) 

async def queryRLSimu(uri: str, m: RLQuery, onCompletion: Callable[[RLResult],None]):
    async with websockets.connect(uri, max_size=2**25, ping_timeout=None) as websocket:
        await websocket.send(serializeRLQueryQuery(m))
        rv = await websocket.recv()       
        r = JSONSerializer.deserialize(RLResult, json.loads(rv))
        #print(r)
        print(r.Status)
        onCompletion(r)

def performQuery(uri: str, q: MarkowitzProblem, onCompletion: Callable[[makeMarkowitzSolution],None]):
    asyncio.get_event_loop().run_until_complete(query(uri, q, onCompletion))

def perforQueryAny(uri: str, q: Dict[str,Any], n: str, onCompletion: Callable[[Dict[str,Any]],None]):
    asyncio.get_event_loop().run_until_complete(queryAny(uri, q, n, onCompletion))

def performQuerySolverProblem(uri: str, m: SolverProblem, onCompletion: Callable[[SolverSolution],None]):
    asyncio.get_event_loop().run_until_complete(querySolverProblem(uri, m, onCompletion))

def performQueryExtraQUBO(uri: str, m: ExtractQUBO, onCompletion: Callable[[QUBOSolution],None]):
    asyncio.get_event_loop().run_until_complete(queryExtraQUBO(uri, m, onCompletion))

def performQueryRLQuery(uri: str, m: RLQuery, onCompletion: Callable[[RLResult],None]):
    asyncio.get_event_loop().run_until_complete(queryRLSimu(uri, m, onCompletion))