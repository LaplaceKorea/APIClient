
from typing import Any, Callable, Dict, List, TYPE_CHECKING, Tuple, Union, cast
import numpy as np

_operatorsIotaCounter = 0
def _operatorsIota() -> int:
    global _operatorsIotaCounter
    rv = _operatorsIotaCounter
    _operatorsIotaCounter = _operatorsIotaCounter+1
    return rv

TERM_ATOM=_operatorsIota()
TERM_ADD=_operatorsIota()
TERM_SUB=_operatorsIota()
TERM_MUL=_operatorsIota()
TERM_DIV=_operatorsIota()
TERM_AND=_operatorsIota()
TERM_EQUAL=_operatorsIota()
TERM_GT=_operatorsIota()
TERM_LT=_operatorsIota()
TERM_SYMBOL=_operatorsIota()
TERM_GETITEM=_operatorsIota()
TERM_QUADRATIC=_operatorsIota()
TERM_MAXIMIZE=_operatorsIota()
TERM_MINIMIZE=_operatorsIota()
TERM_MAXIMUM=_operatorsIota()
TERM_MINIMUM=_operatorsIota()
TERM_DECLARE=_operatorsIota()
TERM_LABEL=_operatorsIota()
TERM_OPT=_operatorsIota()
TERM_TARGET=_operatorsIota()
TERM_BITSIZE=_operatorsIota()

_TermAdd_ : Callable[[Any,Any],Any] = lambda x,y: None
_TermSub_ : Callable[[Any,Any],Any] = lambda x,y: None

_TermMul_ : Callable[[Any,Any],Any] = lambda x,y: None
_TermDiv_ : Callable[[Any,Any],Any] = lambda x,y: None

_TermEqual_ : Callable[[Any,Any],Any] = lambda x,y: None
_TermAnd_ : Callable[[Any,Any],Any] = lambda x,y: None
_TermGt_ : Callable[[Any,Any],Any] = lambda x,y: None
_TermLt_ : Callable[[Any,Any],Any] = lambda x,y: None

_TermGetItem_ : Callable[[Any,Any],Any] = lambda x,y: None

_TermQuadratic_ : Callable[[Any,Any],Any] = lambda x,y: None
_Wrapper_ : Callable[[Any],Any] = lambda x: None

class Term:
    def getTag(self)->int:
        pass
    def __add__(self, x)->"Term":
        return cast("Term",_TermAdd_(self, x))
    def __sub__(self, x)->"Term":
        return cast("Term",_TermSub_(self, x))
    def __mul__(self, x)->"Term":
        return cast("Term",_TermMul_(self, x))
    def __div__(self, x)->"Term":
        return cast("Term",_TermDiv_(self, x))        
    def __pow__(self, exponent)->"Term":
        assert(exponent==2)
        return _TermQuadratic_(1, self)
    def __eq__(self, x)->"Term":
        return cast("Term",_TermEqual_(self, x))   
    def __gt__(self, x)->"Term":
        return cast("Term",_TermGt_(self, x)) 
    def __lt__(self, x)->"Term":
        return cast("Term",_TermLt_(self, x)) 
    def __and__(self, x)->"Term":
        return cast("Term", _TermAnd_(self,x))
    def __getitem__(self, x)->"Term":
        return cast("Term", _TermGetItem_(self,x))
    def __radd__(self, x)->"Term":
        return cast("Term",_TermAdd_(x, self))
    def __rsub__(self, x)->"Term":
        return cast("Term",_TermSub_(x, self))
    def __rmul__(self, x)->"Term":
        return cast("Term",_TermMul_(x,self))
    def __rdiv__(self, x)->"Term":
        return cast("Term",_TermDiv_(x,self))        
    def __rpow__(self, exponent)->"Term":
        assert(exponent==2)
        return _TermQuadratic_(1, self)
    def __req__(self, x)->"Term":
        return cast("Term",_TermEqual_(x,self))   
    def __rgt__(self, x)->"Term":
        return cast("Term",_TermGt_(x,self)) 
    def __rlt__(self, x)->"Term":
        return cast("Term",_TermLt_(x,self)) 
    def __rand__(self, x)->"Term":
        return cast("Term", _TermAnd_(x,self))

class TermAtom(Term):
    def getTag(self)->int:
        return TERM_ATOM
    def __init__(self, x):
        self.tag = self.getTag()
        self.atom = x
    def __repr__(self):
        return "TermAtom(" + repr(self.atom) + ")"

class TermSymbol(Term):
    def getTag(self)->int:
        return TERM_SYMBOL
    def __init__(self, name: str):
        self.tag = self.getTag()
        self.name = name
    def __repr__(self):
        return "TermSymbol('" + self.name +"')"

class TermAdd(Term):
    def getTag(self)->int:
        return TERM_ADD
    def __init__(self, x, y):
        self.tag = self.getTag()
        self.left = cast(Term,_Wrapper_(x))
        self.right = cast(Term,_Wrapper_(y))
    def __repr__(self):
        return "TermAdd(" + str(self.left) + "," + str(self.right) + ")"

class TermSub(Term):
    def getTag(self)->int:
        return TERM_SUB
    def __init__(self, x, y):
        self.tag = self.getTag()
        self.left = cast(Term,_Wrapper_(x))
        self.right = cast(Term,_Wrapper_(y))
    def __repr__(self):
        return "TermSub(" + str(self.left) + "," + str(self.right) + ")"

class TermMul(Term):
    def getTag(self)->int:
        return TERM_MUL
    def __init__(self, x, y):
        self.tag = self.getTag()
        self.left = cast(Term,_Wrapper_(x))
        self.right = cast(Term,_Wrapper_(y))
    def __repr__(self):
        return "TermMul(" + str(self.left) + "," + str(self.right) + ")"

class TermDiv(Term):
    def getTag(self)->int:
        return TERM_DIV
    def __init__(self, x, y):
        self.tag = self.getTag()
        self.left = cast(Term,_Wrapper_(x))
        self.right = cast(Term,_Wrapper_(y))
    def __repr__(self):
        return "TermDiv(" + str(self.left) + "," + str(self.right) + ")"

class TermEqual(Term):
    def getTag(self)->int:
        return TERM_EQUAL
    def __init__(self, x, y):
        self.tag = self.getTag()
        self.left = cast(Term,_Wrapper_(x))
        self.right = cast(Term,_Wrapper_(y))
    def __repr__(self):
        return "TermEqual(" + str(self.left) + "," + str(self.right) + ")"

class TermGt(Term):
    def getTag(self)->int:
        return TERM_GT
    def __init__(self, x, y):
        self.tag = self.getTag()
        self.left = cast(Term,_Wrapper_(x))
        self.right = cast(Term,_Wrapper_(y))
    def __repr__(self):
        return "TermGt(" + str(self.left) + "," + str(self.right) + ")"

class TermLt(Term):
    def getTag(self)->int:
        return TERM_LT
    def __init__(self, x, y):
        self.tag = self.getTag()
        self.left = cast(Term,_Wrapper_(x))
        self.right = cast(Term,_Wrapper_(y))
    def __repr__(self):
        return "TermLt(" + str(self.left) + "," + str(self.right) + ")"

class TermAnd(Term):
    def getTag(self)->int:
        return TERM_AND
    def __init__(self, x, y):
        self.tag = self.getTag()
        self.left = cast(Term,_Wrapper_(x))
        self.right = cast(Term,_Wrapper_(y))
    def __repr__(self):
        return "TermAnd(" + str(self.left) + "," + str(self.right) + ")"

class TermGetItem(Term):
    def getTag(self)->int:
        return TERM_GETITEM
    def __init__(self, x, y):
        self.tag = self.getTag()
        self.left = cast(Term,_Wrapper_(x))
        self.right = cast(Term,_Wrapper_(y))
    def __repr__(self):
        return "TermGetItem(" + str(self.left) + "," + str(self.right) + ")"

class TermDeclare(Term):
    def getTag(self)->int:
        return TERM_DECLARE
    def __init__(self, x, y):
        self.tag = self.getTag()
        self.left = cast(Term,_Wrapper_(x))
        self.right = cast(Term,_Wrapper_(y))
    def __repr__(self):
        return "TermDeclare(" + str(self.left) + "," + str(self.right) + ")"

class TermBitsize(Term):
    def getTag(self)->int:
        return TERM_BITSIZE
    def __init__(self, x, y):
        self.tag = self.getTag()
        self.left = cast(Term,_Wrapper_(x))
        self.right = cast(Term,_Wrapper_(y))
    def __repr__(self):
        return "TermBitsize(" + str(self.left) + "," + str(self.right) + ")"

class TermLabel(Term):
    def getTag(self)->int:
        return TERM_LABEL
    def __init__(self, x, y):
        self.tag = self.getTag()
        self.left = cast(Term,_Wrapper_(x))
        self.right = cast(Term,_Wrapper_(y))
    def __repr__(self):
        return "TermLabel(" + str(self.left) + "," + str(self.right) + ")"

class TermQuadratic(Term):
    def getTag(self)->int:
        return TERM_QUADRATIC
    def __init__(self, m, x):
        self.tag = self.getTag()
        self.m = cast(Term,_Wrapper_(m))
        self.x = cast(Term,_Wrapper_(x))
    def __repr__(self):
        return "TermQuadratic(" + str(self.m) + "," + str(self.x) + ")"    

class TermMaximize(Term):
    def getTag(self)->int:
        return TERM_MAXIMIZE
    def __init__(self, m, x):
        self.tag = self.getTag()
        self.m = cast(Term,_Wrapper_(m))
        self.x = cast(Term,_Wrapper_(x))
    def __repr__(self):
        return "TermMaximize(" + str(self.m) + "," + str(self.x) + ")"       

class TermMinimize(Term):
    def getTag(self)->int:
        return TERM_MINIMIZE
    def __init__(self, m, x):
        self.tag = self.getTag()
        self.m = cast(Term,_Wrapper_(m))
        self.x = cast(Term,_Wrapper_(x))
    def __repr__(self):
        return "TermMinimize(" + str(self.m) + "," + str(self.x) + ")"   

class TermMaximum(Term):
    def getTag(self)->int:
        return TERM_MAXIMUM
    def __init__(self, x):
        self.tag = self.getTag()
        self.x = cast(Term,_Wrapper_(x))
    def __repr__(self):
        return "TermMaximum(" + str(self.x) + ")"       

class TermMinimum(Term):
    def getTag(self)->int:
        return TERM_MINIMUM
    def __init__(self, x):
        self.tag = self.getTag()
        self.x = cast(Term,_Wrapper_(x))
    def __repr__(self):
        return "TermMinimum(" + str(self.x) + ")"   

OPT_TERM_EQ=_operatorsIota()
OPT_TERM_GT=_operatorsIota()
OPT_TERM_LT=_operatorsIota()
OPT_TERM_MINIMIZE=_operatorsIota()
OPT_TERM_MAXIMIZE=_operatorsIota()
OPT_TERM_GT_MIN=_operatorsIota()
OPT_TERM_LT_MIN=_operatorsIota()
OPT_TERM_GT_MAX=_operatorsIota()
OPT_TERM_LT_MAX=_operatorsIota()

# operator, label, quad/lin part
class OptTerm(Term):
    def getTag(self)->int:
        return TERM_OPT    
    def __init__(self, quad, lin, op, cmp, label: str):
        self.quad = quad
        self.lin = lin
        self.op = op
        self.cmp = cmp
        self.label = label
    def __repr__(self):
        def opStr(op: int) -> str:
            if op == OPT_TERM_EQ:
                return "OPT_TERM_EQ"
            if op == OPT_TERM_GT:
                return "OPT_TERM_GT"
            if op == OPT_TERM_LT:
                return "OPT_TERM_LT"
            if op == OPT_TERM_MINIMIZE:
                return "OPT_TERM_MINIMIZE"
            if op == OPT_TERM_MAXIMIZE:
                return "OPT_TERM_MAXIMIZE"
            if op == OPT_TERM_GT_MIN:
                return "OPT_TERM_GT_MIN"
            if op == OPT_TERM_LT_MIN:
                return "OPT_TERM_LT_MIN"
            if op == OPT_TERM_GT_MAX:
                return "OPT_TERM_GT_MAX"
            if op == OPT_TERM_LT_MAX:
                return "OPT_TERM_LT_MAX"
            return str(op)
        return "OptTerm(" + repr(self.quad) + "," + repr(self.lin) + "," + opStr(self.op) + "," + repr(self.cmp) + "," + repr(self.label) + ")"

# Opt terms & declare & weight declares
class Target(Term):
    def getTag(self)->int:
        return TERM_TARGET    
    def __init__(self, components: List[OptTerm], annotations: List[Term], positions: Dict[str,int], sizes: Dict[str,int]):
        self.components = components
        self.annotations = annotations
        self.positions = positions
        self.sizes = sizes
    def __repr__(self):
        return "Target(" + repr(self.components) + "," + repr(self.annotations) + "," + repr(self.positions) + "," + repr(self.sizes) + ")"

# Links
_TermAdd_ = TermAdd
_TermSub_ = TermSub
_TermMul_ = TermMul
_TermDiv_ = TermDiv
_TermAnd_ = TermAnd
_TermEqual_ = TermEqual
_TermGt_ = TermGt
_TermLt_ = TermLt
_TermQuadratic_ = TermQuadratic
_TermGetItem_ = TermGetItem

# helpers
def sym(x:str)->TermSymbol:
    return TermSymbol(x)
def wrap(x: Any)->Term:
    if isinstance(x, int):
        return TermAtom(x)
    if isinstance(x, float):
        return TermAtom(x) 
    if isinstance(x, np.float32):
        return TermAtom(x)      
    if isinstance(x, np.float64):
        return TermAtom(x)
    if isinstance(x, np.ndarray):
        return TermAtom(x)           
    if isinstance(x, str):
        return TermSymbol(x)
    return x

def quadratic(m: Any, x: Any)->Term:
    return TermQuadratic(m, x)

def maximize(v,expr):
    return TermMaximize(v,expr)

def minimize(v,expr):
    return TermMinimize(v,expr)

def sum(expr):
    return wrap(1)*expr

def minimum(expr):
    return TermMinimum(expr)

def maximum(expr):
    return TermMaximum(expr)

def declare(a,b):
    return TermDeclare(a,b)

def label(a,b):
    return TermLabel(a,b)

def bitsize(a,b):
    return TermBitsize(a,b)

_Wrapper_ = wrap

class PositionsHints:
    def __init__(self):
        self.at : Dict[str, int] = {}
        self.inside : Dict[str, str] = {}
        self.minSize : Dict[str, int] = {}
    def __repr__(self):
        return ("PositionHints("  
            + repr(self.at) + ","
            + repr(self.inside) + ","
            + repr(self.minSize)
            + ")")
    def getProblemSize(self):
        mm = 0
        for k in self.at:
            mm = max(self.at[k] + self.minSize[k], mm) # ex: at 0, size=1 => 1 elt
        return mm

def estimateShapeOfExpression(h: PositionsHints, t: Term) -> Tuple[int,int]:
    tag = t.getTag()
    if tag == TERM_ATOM:
        ta = cast(TermAtom, t)
        a = ta.atom
        if isinstance(a, np.ndarray):
            s = a.shape
            if len(s) == 2:
                return s
            return (s[0],1)
        return (1,1)
    if tag == TERM_SYMBOL:
        ts = cast(TermSymbol, t)
        s = ts.name
        if s in h.minSize:
            return (h.minSize[s],1)
        return (1,1)
    if tag == TERM_ADD:
        tadd = cast(TermAdd, t)
        s1 = estimateShapeOfExpression(h, tadd.left)
        s2 = estimateShapeOfExpression(h, tadd.right)
        return (max(s1[0],s2[0]), max(s1[1],s2[1]))
    if tag == TERM_SUB:
        tsub = cast(TermSub, t)
        s1 = estimateShapeOfExpression(h, tsub.left)
        s2 = estimateShapeOfExpression(h, tsub.right)
        return (max(s1[0],s2[0]), max(s1[1],s2[1]))
    if tag == TERM_MUL:
        # convention
        # vector * vector => scalar
        # v * matrix or matrix *v => matrix prod
        tmul = cast(TermMul, t)
        s1 = estimateShapeOfExpression(h, tmul.left)
        s2 = estimateShapeOfExpression(h, tmul.right)
        s1v = s1[1] == 1
        s2v = s2[1] == 1
        #print(s1v,s2v)
        if s1v and s2v:
            # vector/vector => scalar => 1
            return (1,1)
        if not(s1v) and s2v:
            # matrix * vector: vector(lines(matrix))
            return (s1[0],1)
        if s1v and not(s2v):
            # vector transpose * matrix => vector(cols(matrix))
            return (s2[1],1)
        if not(s1v) and not(s2v):
            # matrix matrix
            return (s1[0], s2[1])
        return (1,1)
    if tag == TERM_QUADRATIC:
        return (1,1)
    return (1,1)

def propagateShape(h: PositionsHints, t: Term, sh: Tuple[int,int]):
    tag = t.getTag()
    if tag == TERM_ATOM:
        ta = cast(TermAtom, t)
        a = ta.atom
        if isinstance(a, np.ndarray):
            s = a.shape
            if len(s) == 2:
                if s == (1,1):
                    v = s[0,0]
                    a = np.diag([v for k in range(sh[0])], dtype=np.float64)
                    a = ta.atom
                    #print(" * resize matrix", a.shape)
            else:
                if sh[0] > s[0]:
                    v = s[0]
                    a = np.array([v for k in range(sh[0])])
                    ta.atom = a
                    #print("resize matrix", a.shape)
        if isinstance(a, int) or isinstance(a, float) or isinstance(a, np.float32) or isinstance(a, np.float64):
            if sh[1]==1:
                a = np.array([a for k in range(sh[0])], dtype=np.float64)
                ta.atom = a
                #print("resize scalar to vector", a.shape)
            else:
                a = np.diag([a for k in range(sh[0])], dtype=np.float64)
                ta.atom = a
                #print("resize scalar to diag matrix", a.shape)
        pass
    if tag == TERM_SYMBOL:
        ts = cast(TermSymbol, t)
        s = ts.name
        if s in h.minSize:
            _s2 = h.minSize[s]
            _sf = max(sh[0], _s2)
            h.minSize[s] = _sf
            #print("symbol",s,"resized to",_sf)
        else:
            h.minSize[s] = sh[0]
            #print("symbol",s,"acquired size",sh[0])
    if tag == TERM_ADD:
        tadd = cast(TermAdd, t)
        s1 = estimateShapeOfExpression(h, tadd.left)
        s2 = estimateShapeOfExpression(h, tadd.right)
        sf = (max(sh[0],max(s1[0],s2[0])), max(sh[1],max(s1[1],s2[1])))
        propagateShape(h, tadd.left, sh)
        propagateShape(h, tadd.right, sh)   
    if tag == TERM_SUB:
        tsub = cast(TermSub, t)
        s1 = estimateShapeOfExpression(h, tsub.left)
        s2 = estimateShapeOfExpression(h, tsub.right)
        sf = (max(sh[0],max(s1[0],s2[0])), max(sh[1],max(s1[1],s2[1])))
        propagateShape(h, tsub.left, sh)
        propagateShape(h, tsub.right, sh) 
    if tag == TERM_MUL:
        # convention
        # vector * vector => scalar
        # v * matrix or matrix *v => matrix prod
        tmul = cast(TermMul, t)
        s1 = estimateShapeOfExpression(h, tmul.left)
        s2 = estimateShapeOfExpression(h, tmul.right)
        s1v = s1[1] == 1
        s2v = s2[1] == 1
        #print(s1v,s2v)
        if s1v and s2v:
            # vector/vector => scalar => 1
            # s1 and s2 must be equals~
            sf = (max(sh[0],max(s1[0],s2[0])), 1)
            propagateShape(h, tmul.left, sf)
            propagateShape(h, tmul.right, sf) 
        if not(s1v) and s2v:
            # matrix * vector: vector(lines(matrix))
            ## return (s1[0],1)
            sz = max(s1[0], sh[0])
            propagateShape(h,tmul.left,(sz,s1[1]))
        if s1v and not(s2v):
            # vector transpose * matrix => vector(cols(matrix))
            ## return (s2[1],1)
            sz = max(s2[1], sh[0])
            propagateShape(h,tmul.right, (sz,s2[1]))
        if not(s1v) and not(s2v):
            # matrix matrix
            ## return (s1[0], s2[1])
            sz1 = max(s1[0], sh[0])
            sz2 = max(s2[1], sh[1])
            propagateShape(h, tmul.left, (sz1, s1[1]))
            propagateShape(h, tmul.right, (s2[0], sz2))
        pass
    if tag == TERM_QUADRATIC:
        tq = cast(TermQuadratic, t)
        shM = estimateShapeOfExpression(h,tq.m)
        shX = estimateShapeOfExpression(h,tq.x)
        sz = max(shM[0], shX[0])
        propagateShape(h, tq.m, (sz,sz))
        propagateShape(h, tq.x, (sz,1))
    # propagate to other types
    if tag == TERM_EQUAL:
        te = cast(TermEqual, t)
        s1 = estimateShapeOfExpression(h, te.left)
        s2 = estimateShapeOfExpression(h, te.right)
        sf = (max(sh[0],max(s1[0],s2[0])), max(sh[1],max(s1[1],s2[1])))
        propagateShape(h, te.left, sh)
        propagateShape(h, te.right, sh)   
    if tag == TERM_LT:
        tlt = cast(TermLt, t)
        s1 = estimateShapeOfExpression(h, tlt.left)
        s2 = estimateShapeOfExpression(h, tlt.right)
        sf = (max(sh[0],max(s1[0],s2[0])), max(sh[1],max(s1[1],s2[1])))
        propagateShape(h, tlt.left, sh)
        propagateShape(h, tlt.right, sh)
    if tag == TERM_GT:
        tgt = cast(TermGt, t)
        s1 = estimateShapeOfExpression(h, tgt.left)
        s2 = estimateShapeOfExpression(h, tgt.right)
        sf = (max(sh[0],max(s1[0],s2[0])), max(sh[1],max(s1[1],s2[1])))
        propagateShape(h, tgt.left, sh)
        propagateShape(h, tgt.right, sh)
    if tag == TERM_AND:
        tand = cast(TermAnd, t)
        propagateShape(h, tand.left, (1,1))
        propagateShape(h, tand.right, (1,1))
    if tag == TERM_GETITEM:
        tget = cast(TermGetItem, t)
        propagateShape(h, tget.left, (1,1))
        propagateShape(h, tget.right, (1,1))
        if tget.left.getTag() == TERM_SYMBOL and tget.right.getTag() == TERM_SYMBOL:
            sym1 = cast(TermSymbol, tget.left)
            sym2 = cast(TermSymbol, tget.right)
            h.inside[sym2.name] = sym1.name
    if tag == TERM_LABEL:
        tlabel = cast(TermLabel, t)
        # NO! propagateShape(h, tlabel.left, (1,1))
        propagateShape(h, tlabel.right, sh) #!!!! 
    if tag == TERM_MAXIMIZE:
        tmaxi = cast(TermMaximize, t)
        ##propagateShape(h, tmaxi.m, (1,1))
        propagateShape(h, tmaxi.x, (1,1))
    if tag == TERM_MINIMIZE:
        tmini = cast(TermMinimize, t)
        ##propagateShape(h, tmini.m, (1,1))
        propagateShape(h, tmini.x, (1,1))
    if tag == TERM_DECLARE:
        tdecl = cast(TermDeclare, t)
        if tdecl.left.getTag() == TERM_SYMBOL and tdecl.right.getTag() == TERM_ATOM:
            sym1 = cast(TermSymbol, tdecl.left)
            at1 = cast(TermAtom, tdecl.right)
            h.at[sym1.name] = int(at1.atom)
    pass

def fillSlots(sizes: Dict[str,int], positions: Dict[str, int])->List[str]:
    rv: List[str] = []
    def getAt(x: int)->str:
        if len(rv) > x:
            return rv[x]
        else:
            for i in range(x - len(rv) + 1):
                rv.append("") # => void
            return ""
    def setAt(x: int, v: str):
        if len(rv) > x:
            rv[x] = v
        else:
            for i in range(x - len(rv) + 1):
                rv.append("") # => void
            rv[x] = v        
    for p in positions:
        setAt(positions[p], p)
        if sizes[p] > 1:
            for i in range(sizes[p]-1):
                setAt(positions[p]+1, "#")
    szs: List[Tuple[str,int]] = []
    for k in sizes:
        szs.append((k, sizes[k]))
    szs.sort(key = lambda x:x[1])
    #print("sorted", szs)
    def findPosition(n: str, s:int):
        if n in positions:
            return positions[n]
        cursor = 0
        failedAt = 0
        def isCursorOk() -> bool:
            nonlocal cursor
            nonlocal failedAt
            failedAt = 0
            for i in range(s):
                if getAt(cursor + i) != "":
                    failedAt = i
                    #print("failed @ ", i)
                    return False
            #print("accept, cursor = ", cursor)
            return True
        while not isCursorOk():
            cursor = cursor + failedAt+1
            #print("cursor updated = ", cursor)
        return cursor
    for i in range(len(szs)):
        #print("before insert, rv=", rv)
        n = szs[len(szs)-1-i][0]
        size = szs[len(szs)-1-i][1]
        pos = findPosition(n, size)
        positions[n] = pos
        setAt(pos, n)
        if sizes[n] > 1:
            for i in range(sizes[n]-1):
                setAt(pos+1+i, "#")
        #print("rv= ", rv)
        #print(n,size,"@",pos)
    #print("var map ", rv)
    return rv

def findSizeAndPositions(h: PositionsHints, t: Term):
    propagateShape(h, t, (1,1))
    propagateShape(h, t, (1,1))
    #print(">>", h)
    hints: Dict[str,int] = {}
    pos: Dict[str, int] = {}
    for k in h.minSize:
        if k in h.inside:
            pass
        else:
            hints[k] = h.minSize[k]
            if k in h.at:
                pos[k] = h.at[k]
    #print("fillSlots", hints, pos)
    fillSlots(hints, pos)
    for k in pos:
        h.at[k] = pos[k]
    for k in hints:
        h2: Dict[str,int] = {}
        p2: Dict[str,int] = {}
        for p in h.inside:
            if h.inside[p] == k:
                h2[p] = h.minSize[p]
                if p in h.at:
                    p2[p] = h.at[p]
        if len(h2) > 0:
            fillSlots(h2,p2)
            for p in p2:
                h.at[p] = p2[p]

def extractSquareMatrix(h: PositionsHints, t: Term, windowBase: int, windowSize:int)->Union[None,np.ndarray]:
    tag = t.getTag()
    sz = h.getProblemSize()
    if tag==TERM_ATOM:
        ta=cast(TermAtom, t)
        atom=ta.atom
        if isinstance(atom, int):
            return np.array([[float(atom)]], dtype=np.float64)
        if isinstance(atom, float):
            return np.array([[float(atom)]], dtype=np.float64)
        if isinstance(atom, np.ndarray):
            if atom.shape == (1,):
                return atom
            if len(atom.shape) == 2:
                return atom
    return None

def symmetrize(m: np.ndarray)->np.ndarray:
    return 0.5*(m+np.transpose(m))

# Affine
def extractMultiDimLinearForm(h: PositionsHints, t: Term, windowBase: int, windowSize:int)->Tuple[np.ndarray, np.ndarray]:
    #print("extraMultiDimLinearForm", h, t, windowBase, windowSize)
    tag = t.getTag()
    sz = h.getProblemSize()
    if tag == TERM_SYMBOL:
        ts = cast(TermSymbol, t)
        n = ts.name
        pos = h.at[n]
        lsz = h.minSize[n]
        if pos >= windowBase + windowSize: ## TODO or .... pos < (and for other places like that) 
            # no intersect
            return (np.zeros((sz,sz), dtype=np.float64), np.array((sz,), dtype=np.float64))
        linear = np.zeros((sz,sz), dtype=np.float64)
        for i in range(lsz):
            if i+pos>=windowBase and i < windowBase+windowSize:
                linear[i+pos,i+pos] = 1.0
        return (linear, np.zeros((sz,), dtype=np.float64))
    if tag == TERM_MUL:
        tmul = cast(TermMul, t)
        mm1 = extractSquareMatrix(h, tmul.left, windowBase, windowSize)
        mm2 = extractSquareMatrix(h, tmul.right, windowBase, windowSize)

        def sizeUp(mm, Ax):
            #print("sizeUp", mm, Ax)
            # find indices, copy
            tgtSq = np.zeros((sz, sz),dtype=np.float64)
            hasLines = []
            for i in range(sz):
                hasLine = False
                for j in range(sz):
                    if abs(Ax[i,j]) > 1e-12:
                        hasLine = True
                # NO! if abs(Bx[i]) > 1e-12:
                #    hasLine = True
                hasLines.append(hasLine)
            #print("hasLines =", hasLines)
            cursori = 0
            for i in range(sz):
                cursorj = 0
                for j in range(sz):
                    if hasLines[i] and hasLines[j]:
                        tgtSq[i,j] = mm[cursori, cursorj]
                    if hasLines[j]:
                        cursorj = cursorj+1
                if hasLines[i]:
                    cursori = cursori+1
            #print(tgtSq)
            return tgtSq
        (Ax, Bx) = (None, None)
        try:
            (Ax, Bx) = extractMultiDimLinearForm(h, tmul.left, windowBase, windowSize)
        except:
            pass
        (Ay, By) = (None, None)
        try:
            (Ay, By) = extractMultiDimLinearForm(h, tmul.right, windowBase, windowSize)
        except:
            pass
        if mm1 is None:
            if mm2 is None:
                # last resort/desperate and of course false in general
                # (Ax,Bx) (x) (Ay,By) ~ By^t (Ax,Bx) + Bx^t (Ay,By) + Bx .. By
                # will it even work?
                return (np.matmul(np.transpose(By),Ax) + np.matmul(np.transpose(Bx),Ay), np.matmul(np.transpose(Bx), By))
            else:
                # (Ax,Bx) (x) matrix ~ matrix^t * (Ax,Bx)
                mm2_ = sizeUp(mm2, Ax)
                mm2t = np.transpose(mm2_)
                ##print("mul/757", mm2t, Ax, Bx)
                return (np.matmul(mm2t,Ax), np.matmul(mm2t, Bx))
        else:
            if mm2 is None:
                mm1_ = sizeUp(mm1, Ay)
                ##print("mul/785", mm1_, Ay, By)
                return (np.matmul(mm1_, Ay), np.matmul(mm1_,By))
            else:
                # panic!
                raise Exception("mat * mat => non linear")
    if tag == TERM_SUB:
        tsub = cast(TermSub, t)
        (Ax, Bx) = extractMultiDimLinearForm(h, tsub.left, windowBase, windowSize)
        (Ay, By) = extractMultiDimLinearForm(h, tsub.right, windowBase, windowSize)
        return (Ax-Ay, Bx-By)
    if tag == TERM_ADD:
        tadd = cast(TermAdd, t)
        (Ax, Bx) = extractMultiDimLinearForm(h, tadd.left, windowBase, windowSize)
        (Ay, By) = extractMultiDimLinearForm(h, tadd.right, windowBase, windowSize)
        return (Ax+Ay, Bx+By)
    if tag == TERM_ATOM:
        tatom = cast(TermAtom, t)
        atom = tatom.atom
        #print("804", atom)
        if isinstance(atom, int):
            return (np.zeros((sz,sz), dtype=np.float64), np.array([float(atom) for i in range(sz)], dtype=np.float64))
        if isinstance(atom, float):
            return (np.zeros((sz,sz), dtype=np.float64), np.array([float(atom) for i in range(sz)], dtype=np.float64))
        if isinstance(atom, np.ndarray):
            # here, better be sure that you are at 0... no time for it at this stage ~~~~
            if len(atom.shape) == 2:
                raise Exception("lin terms only here!")
            rv = np.zeros((sz,), dtype=np.float64)
            for i in range(atom.shape[0]):
                rv[i] = atom[i]
            if (sz > atom.shape[0]): # UGLY, needed for expressions like (x-1)**2 + (y-2)**2, LT solution is to add a /range/ info the the args
                for i in range(sz - atom.shape[0]):
                    rv[i+atom.shape[0]] = atom[0]
            return (np.zeros((sz,sz), dtype=np.float64), rv)
    raise Exception("Term type not supported")

def extractQuadraticForm(h: PositionsHints, t: Term, windowBase: int, windowSize:int)->Tuple[np.ndarray,np.ndarray,float]:
    tag = t.getTag()
    sz = h.getProblemSize()
    if tag == TERM_SYMBOL:
        ts = cast(TermSymbol, t)
        n = ts.name
        pos = h.at[n]
        lsz = h.minSize[n]
        if pos >= windowBase + windowSize:
            # no intersect
            return (np.zeros((sz,sz), dtype=np.float64), np.array((sz,), dtype=np.float64), 0.0)
        linear = np.zeros((sz,), dtype=np.float64)
        quad = np.zeros((sz,sz), dtype=np.float64)
        for i in range(lsz):
            if pos+i>=windowBase and pos+i < windowBase+windowSize:
                linear[i+pos] = 1.0
        return (quad, linear, 0.0)
    if tag == TERM_ADD:
        ta = cast(TermAdd, t)
        q1,l1,s1 = extractQuadraticForm(h, ta.left, windowBase, windowSize)
        q2,l2,s2 = extractQuadraticForm(h, ta.right, windowBase, windowSize)
        return (symmetrize(q1+q2), l1+l2, s1+s2)
    if tag == TERM_SUB:
        tsub = cast(TermSub, t)
        q1,l1,s1 = extractQuadraticForm(h, tsub.left, windowBase, windowSize)
        q2,l2,s2 = extractQuadraticForm(h, tsub.right, windowBase, windowSize)
        return (symmetrize(q1-q2), l1-l2, s1-s2)
    if tag == TERM_MUL:
        tm = cast(TermMul, t)
        try:
            q1,l1,s1 = extractQuadraticForm(h, tm.left, windowBase, windowSize)
            q2,l2,s2 = extractQuadraticForm(h, tm.right, windowBase, windowSize)
            # notice this makes quad * quad = 0 [sorry for this version: this is a demo]
            #print("quadratic mul:", (q1,l1,s1), (q2,l2,s2))

            debug1 = np.matmul(np.transpose(np.reshape(l1, (l1.shape[0],1))),np.reshape(l2,(l2.shape[0],1)))
            debug2 = np.transpose(np.reshape(l1, (l1.shape[0],1))),np.reshape(l2,(l2.shape[0],1))
            #print("debug", debug1, debug2)

            return (symmetrize(s1 * q2 + s2 * q2 + np.matmul(np.reshape(l1, (l1.shape[0],1)),np.transpose(np.reshape(l2,(l2.shape[0],1))))), l1*s2 + l2*s1, s1*s2)
        except:
            A1, B1 = extractMultiDimLinearForm(h, tm.left, windowBase, windowSize)
            A2, B2 = extractMultiDimLinearForm(h, tm.right, windowBase, windowSize)
            #print("MUL", (A1,B1), (A2,B2))
            return (symmetrize(np.matmul(np.transpose(A1),A2)), np.matmul(np.transpose(B1), A2) + np.matmul(np.transpose(B2), A1), np.dot(B1,B2))
    if tag == TERM_ATOM:
        tatom = cast(TermAtom, t)
        # refute non-scalar quantities
        a = tatom.atom
        if isinstance(a,int):
            return (np.zeros((sz,sz), dtype=np.float64), np.zeros((sz,), dtype=np.float64), float(a))
        if isinstance(a,float):
            return (np.zeros((sz,sz), dtype=np.float64), np.zeros((sz,), dtype=np.float64), float(a))
        if isinstance(a,np.ndarray):
            if a.shape == (1,):
                return (np.zeros((sz,sz), dtype=np.float64), np.zeros((sz,), dtype=np.float64), float(a[0]))
            if a.shape != (1,1):
                #print("to scalar?", a)
                raise Exception("atom is not trivially castable to scalar")
            return (np.zeros((sz,sz), dtype=np.float64), np.zeros((sz,), dtype=np.float64), float(a[0,0]))
    if tag == TERM_QUADRATIC:
        tquad = cast(TermQuadratic, t) # TODO symmetrize matrices!!!! ~ 
        m = tquad.m
        x = tquad.x
        # N.B. again will not work on all cases. Pray that only lx is non-zero here!
        (Ax, Bx) = extractMultiDimLinearForm(h, x, windowBase, windowSize)
        mm = extractSquareMatrix(h, m, windowBase, windowSize)
        #print("debug, term quadratic", t, (Ax,Bx,mm))
        if mm is None:
            raise Exception("no matrix!")
        else:
            mm = symmetrize(mm)
        # find indices, copy
        tgtSq = np.zeros((sz, sz),dtype=np.float64)
        # special case: [1]*x**2
        if mm.shape == (1,):
            for i in range(sz):
                tgtSq[i,i] = mm[0]
        else:
            hasLines = []
            for i in range(sz):
                hasLine = False
                for j in range(sz):
                    if abs(Ax[i,j]) > 1e-12:
                        hasLine = True
                # NO! if abs(Bx[i]) > 1e-12:
                #    hasLine = True
                hasLines.append(hasLine)
            #print("haslines = ", hasLines)
            cursori = 0
            for i in range(sz):
                cursorj = 0
                for j in range(sz):
                    if hasLines[j] and hasLines[i]:
                        tgtSq[i,j] = mm[cursori, cursorj]
                    if hasLines[j]:
                        cursorj = cursorj+1
                if hasLines[i]:
                    cursori = cursori+1
        #print("debug/tq", tgtSq, Ax, Bx)
        return (np.matmul(np.transpose(Ax),np.matmul(tgtSq,Ax)), 2*np.matmul(np.transpose(Bx),np.matmul(tgtSq,Ax)), np.dot(Bx,Bx))
    if tag == TERM_GETITEM:
        tgi = cast(TermGetItem, t)
        #print("> getItem", tgi)
        assert(tgi.right.getTag() == TERM_SYMBOL)
        return extractQuadraticForm(h, tgi.right, windowBase, windowSize) # again some approx here but...
    #print(t)
    raise Exception("node type not supported for extracting quadratic form")

def extractOptTermOrCondition(t: Term) -> Target:
    def extractConst(t: Term, h: PositionsHints, quad: np.ndarray, lin: np.ndarray) -> np.ndarray:
        sz = h.getProblemSize()
        tag = t.getTag()
        if tag == TERM_ATOM:
            tatom = cast(TermAtom, t)
            atom = tatom.atom
            if isinstance(atom, int) or isinstance(atom, int):
                rv = np.zeros((sz,))
                for i in range(sz):
                    has_elt = False
                    for j in range(sz):
                        if abs(quad[i,j]) > 1e-12:
                            has_elt = True
                    if abs(lin[i]) > 1e-12:
                        has_elt = True
                    if has_elt:
                        rv[i] = float(atom)
                return rv
            if isinstance(atom, np.ndarray):
                if atom.shape[0] == sz:
                    return atom
                else:
                    #print("extract const>", atom.shape, atom, sz, quad, lin)
                    cursor = 0
                    rv = np.zeros((sz,))
                    for i in range(sz):
                        has_elt = False
                        for j in range(sz):
                            if abs(quad[i,j]) > 1e-12:
                                has_elt = True
                        if abs(lin[i]) > 1e-12:
                            has_elt = True
                        if has_elt:
                            rv[i] = atom[cursor] if atom.shape[0] != 1 else atom[0] # dirty
                            cursor = cursor + 1
                    return rv                    
        raise Exception("cannot extract const")
    def extraOptTermOrCondition_(t: Term, ta: Target, h: PositionsHints) -> Target:
        tag = t.getTag()
        #print("extraOptT/C", t, tag, TERM_EQUAL)
        if tag == TERM_AND:
            tand = cast(TermAnd, t)
            ta1 = extraOptTermOrCondition_(tand.left, Target([],[], h.at, h.minSize), h)
            ta2 = extraOptTermOrCondition_(tand.right, Target([],[], h.at, h.minSize), h)
            return Target(ta1.components + ta2.components + ta.components,
                            ta1.annotations + ta2.annotations + ta.annotations,
                            h.at, h.minSize)
        if tag == TERM_DECLARE:
            tdecl = cast(TermDeclare, t)
            return Target(ta.components,ta.annotations + [tdecl], h.at, h.minSize)
        if tag == TERM_EQUAL:
            teq = cast(TermEqual, t)
            #print(">>> equ ", t, teq.left)
            quad,lin,const = extractQuadraticForm(h, teq.left, 0, h.getProblemSize())
            o = OptTerm(quad, lin, OPT_TERM_EQ, extractConst(teq.right,h,quad,lin), "")
            return Target(ta.components + [o], ta.annotations, h.at, h.minSize)
        if tag == TERM_GETITEM:
            tgi = cast(TermGetItem, t)
            return Target(ta.components, ta.annotations + [tgi], h.at, h.minSize)
        if tag == TERM_BITSIZE:
            tbs = cast(TermBitsize, t)
            return Target(ta.components, ta.annotations + [tbs], h.at, h.minSize)            
        if tag == TERM_LABEL:
            tlabel = cast(TermLabel, t)
            #print(">> label ", t, tlabel.right)
            tgt1 = extraOptTermOrCondition_(tlabel.right, Target([],[], h.at, h.minSize), h)
            label = tlabel.left
            assert(label.getTag() == TERM_SYMBOL)
            tl = cast(TermSymbol, label)
            #print(">>", tl, tgt1)
            return Target([OptTerm(opt.quad, opt.lin, opt.op, opt.cmp, tl.name if opt.label=="" else opt.label) for opt in tgt1.components] + ta.components, 
                                tgt1.annotations + ta.annotations, h.at, h.minSize)
        if tag == TERM_TARGET:
            ttarget = cast(Target, t)
            return Target(ttarget.components + ta.components, ttarget.annotations + ta.annotations, h.at, h.minSize)
        if tag == TERM_GT:
            tgt = cast(TermGt, t)
            # special case: max/min (expression)
            if (tgt.left.getTag() == TERM_MAXIMUM):
                tmaximum = cast(TermMaximum,tgt.left)
                quad,lin,const = extractQuadraticForm(h, tmaximum.x, 0, h.getProblemSize())
                o = OptTerm(quad, lin, OPT_TERM_GT_MAX, extractConst(tgt.right,h,quad,lin), "")
                return Target(ta.components + [o], ta.annotations, h.at, h.minSize)                
            if (tgt.left.getTag() == TERM_MINIMUM):
                tminimum = cast(TermMinimum, tgt.left)
                quad,lin,const = extractQuadraticForm(h, tminimum.x, 0, h.getProblemSize())
                o = OptTerm(quad, lin, OPT_TERM_GT_MIN, extractConst(tgt.right,h,quad,lin), "")
                return Target(ta.components + [o], ta.annotations, h.at, h.minSize)
            quad,lin,const = extractQuadraticForm(h, tgt.left, 0, h.getProblemSize())
            o = OptTerm(quad, lin, OPT_TERM_GT, extractConst(tgt.right,h,quad,lin), "")
            return Target(ta.components + [o], ta.annotations, h.at, h.minSize)
        if tag == TERM_LT:
            tlt = cast(TermLt, t)
            # special case: max/min (expression)
            if (tlt.left.getTag() == TERM_MAXIMUM):
                tmaximum = cast(TermMaximum,tlt.left)
                quad,lin,const = extractQuadraticForm(h, tmaximum.x, 0, h.getProblemSize())
                o = OptTerm(quad, lin, OPT_TERM_LT_MAX, extractConst(tlt.right,h,quad,lin), "")
                return Target(ta.components + [o], ta.annotations, h.at, h.minSize)                
            if (tlt.left.getTag() == TERM_MINIMUM):
                tminimum = cast(TermMinimum, tlt.left)
                quad,lin,const = extractQuadraticForm(h, tminimum.x, 0, h.getProblemSize())
                o = OptTerm(quad, lin, OPT_TERM_LT_MIN, extractConst(tlt.right,h,quad,lin), "")
                return Target(ta.components + [o], ta.annotations, h.at, h.minSize)
            quad,lin,const = extractQuadraticForm(h, tlt.left, 0, h.getProblemSize())
            o = OptTerm(quad, lin, OPT_TERM_LT, extractConst(tlt.right,h,quad,lin), "")
            return Target(ta.components + [o], ta.annotations, h.at, h.minSize)
        if tag == TERM_MAXIMIZE:
            tmaximize = cast(TermMaximize, t)
            quad,lin,const = extractQuadraticForm(h, tmaximize.x, 0, h.getProblemSize())
            o = OptTerm(quad, lin, OPT_TERM_MAXIMIZE, np.ndarray([0]), "")
            return Target(ta.components + [o], ta.annotations, h.at, h.minSize)
        if tag == TERM_MINIMIZE:
            tminimize = cast(TermMinimize, t)
            quad,lin,const = extractQuadraticForm(h, tminimize.x, 0, h.getProblemSize())
            o = OptTerm(quad, lin, OPT_TERM_MINIMIZE, np.ndarray([0]), "")
            return Target(ta.components + [o], ta.annotations, h.at, h.minSize)
        return ta
    h = PositionsHints()
    findSizeAndPositions(h,t)
    #print("hints for problem", h)
    return extraOptTermOrCondition_(t, Target([],[], h.at, h.minSize), h)

## TODO: eval AST to check end results!
## TODO: sum(.) bug to cleanup ~