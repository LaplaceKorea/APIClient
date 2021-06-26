from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Callable
import numpy as np

@dataclass
class UserTokenSerde:
    user: str
    token: str

@dataclass
class OperationStatusSerde:
    credit: int
    message: str
