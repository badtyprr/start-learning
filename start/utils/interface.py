# Interfacing utilities

# Python Libraries
from typing import Callable, Any
import functools

abstract_function_t = Callable[..., Any]

def deferrableabstractmethod(func: abstract_function_t):
    """
    Python's abc.abstractmethod decorator must be immediately overridden by
    a subclass. Sometimes the method needs to be implemented by an even later
    subclass.

    Suppose we have an inheritance structure for classes A, B, and C:
        A -> B -> C
    A has a function 'func' that will be overridden, but not until class C.
    B will also be an abstract class.

    However, Python cannot yet support this behavior, hence this decorator.
    Upon application of the decorator, the default function behavior is to
    raise a NotImplementedError unless the function is overridden. This
    decorator has a few different behaviors from @abc.abstractmethod:

    * Class B still is a subclass of abc.ABC by inheritance of class A,
        but because deferred abstract methods are not 'registered' with ABC,
        the ABC will allow instantiations of class B, even if the deferred
        abstract methods are not implemented.
    * Class C will also allow instantiation without implementation for
        the same reason as class B.

    :param func: Callable function type representing the function to wrap
    :return: Callable type representing the wrapped function
    """
    functools.wraps(func)
    def wrapper_optionalabstractmethod(*args, **kwargs):
        raise NotImplementedError
    return wrapper_optionalabstractmethod
