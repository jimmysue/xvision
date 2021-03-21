from .nima import NIMA

def initailize_model(module, name, *args, **kwargs):
    return module[name](*args, **kwargs)