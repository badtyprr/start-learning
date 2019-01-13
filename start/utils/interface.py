# Interfacing utilities

# Python Libraries
from typing import Callable, Any
import functools

abstract_function_t = Callable[..., Any]

def deferrableabstractmethod(func: abstract_function_t):
    functools.wraps(func)
    def wrapper_optionalabstractmethod(*args, **kwargs):
        raise NotImplementedError
    return wrapper_optionalabstractmethod
