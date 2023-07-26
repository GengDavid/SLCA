import torch
from models.slca import SLCA

def get_model(model_name, args):
    name = model_name.lower()
    if 'slca' in name:
        return SLCA(args)
    else:
        assert 0
