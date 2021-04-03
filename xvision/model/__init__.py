from .nima import NIMA


def initialize_model(module, name, *args, **kwargs):
    return module.__dict__[name](*args, **kwargs)
