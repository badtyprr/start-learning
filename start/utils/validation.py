# Validation tools

def _validate_types(**kwargs):
    for k, v in kwargs.items():
        if not isinstance(k, v):
            raise ValueError('{} was expected to be of type {}, but got {} instead'.format(k, v, type(k)))
