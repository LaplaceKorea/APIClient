from typing import Callable, Dict, List, Tuple, cast
import Operators as op
import Target as ta
import numpy as np

def toTarget(o: op.Target, weights: Dict[str,float]) -> ta.Target:
    print("toTarget:", o)
    terms : List[ta.Term] = []
    parameters : Dict[str,float] = {}
    init: Dict[str,Tuple[ta.Embedding, np.ndarray]] = {}
    interpretation: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}
    smax = 0
    def makeInterpretation(a:int, b:int) -> Callable[[np.ndarray], np.ndarray]:
        return lambda x: x[a:b]
    for k in o.positions:
        p = o.positions[k]
        s = o.sizes[k] 
        s2 = p+s
        if s2 > smax:
            smax = s2
        interpretation[k] = makeInterpretation(p, s2)
    idMat = np.diag([1.0 for k in range(smax)])
    idEmbedding = ta.Embedding("id", idMat)
    for c in o.components:
        tag = c.op # getTag() is TERM_OPT, always
        if tag == op.OPT_TERM_EQ:
            ntEqu = ta.Equal(c.label, idEmbedding, np.reshape(c.lin, (1, c.lin.shape[0])), np.array([[np.sum(c.cmp)]])) # bad trick
            terms.append(ntEqu)
            parameters[c.label] = 1.0
        if tag == op.OPT_TERM_GT:
            ntGt = ta.GreaterThan(c.label, idEmbedding, np.reshape(c.lin, (1, c.lin.shape[0])), np.array([[np.sum(c.cmp)]])) # bad trick
            terms.append(ntGt)
            parameters[c.label] = 1.0
        if tag == op.OPT_TERM_GT_MAX: # MAX > ...
            raise Exception("GT_MAX not supported for now")
        if tag == op.OPT_TERM_LT:
            ntLt = ta.LessThan(c.label, idEmbedding, np.reshape(c.lin, (1, c.lin.shape[0])), np.array([[np.sum(c.cmp)]])) # bad trick
            terms.append(ntLt)
            parameters[c.label] = 1.0
        if tag == op.OPT_TERM_GT_MIN: # MIN > ... and remove if 0 ~ 
            print("gt_min", c)
            pass
        if tag == op.OPT_TERM_LT_MAX: # MAX < ... => forall x: x < ...
            print("lt_max", c)
            pass
        if tag == op.OPT_TERM_LT_MIN: # MIN < ...
            raise Exception("LT_MIN not supported for now")
        if tag == op.OPT_TERM_MAXIMIZE:
            ntMaximize = ta.Max(c.label, idEmbedding, c.quad, c.lin)
            terms.append(ntMaximize)
            parameters[c.label] = 1.0
        if tag == op.OPT_TERM_MINIMIZE:
            ntMinimize = ta.Min(c.label, idEmbedding, c.quad, c.lin)
            terms.append(ntMinimize)
            parameters[c.label] = 1.0
    bs: List[int] = []
    for a in o.annotations:
        tg = a.getTag()
        if tg == op.TERM_LABEL:
            pass
        if tg == op.TERM_BITSIZE:
            print("bitsize", a)
            tgb = cast(op.TermBitsize, a)
            name = tgb.left
            ns = ""
            if name.getTag() == op.TERM_SYMBOL:
                ns = cast(op.TermSymbol, name).name
            else:
                raise Exception("bitsize/symbol/bad format")
            nb = tgb.right
            nbits = 0
            if nb.getTag() == op.TERM_ATOM:
                nbits = int(cast(op.TermAtom, nb).atom)
            else:
                raise Exception("bitsize/atom/bad format")
            print(ns, nbits)
            idx = o.positions[ns]
            while idx >= len(bs):
                bs.append(0)
            bs[idx] = nbits
    for w in parameters:
        if w in weights:
            print(w, "<-", weights[w])
            parameters[w] = weights[w] 
    return ta.Target(terms, bs, parameters, init, interpretation)
    
def expressionToTarget(o: op.Term, weights: Dict[str,float]) -> ta.Target:
    t = op.extractOptTermOrCondition(o)
    return toTarget(t, weights)
