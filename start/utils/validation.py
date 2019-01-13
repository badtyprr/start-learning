# Validation tools

# Python Packages
from typing import Union, Tuple
# User Packages
from ..utils import _get_union_types

def _validate_types(**kwargs):
    for k, v in kwargs.items():
        try:
            if not isinstance(k, v):
                raise TypeError('{} was expected to be of type {}, but got {} instead'.format(k, v, type(k)))
        except TypeError:
            # Try to handle Union types, which are not compatible directly with isinstance
            found_type = False
            for t in _get_union_types(v):
                if isinstance(k, t):
                    found_type = True
            if not found_type:
                raise TypeError('{} was expected to be of type {}, but got {} instead'.format(k, v, type(k)))

def _get_union_types(u: Union[...]) -> Tuple[...]:
    return u.__args__
