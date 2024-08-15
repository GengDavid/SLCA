import torch
from models.slca import SLCA
from models.slca_pp import SLCApp

def get_model(model_name, args):
    name = model_name.lower()
    if 'slcapp' in name:
        return SLCApp(args)
    elif 'slca' in name:
        return SLCA(args)
    else:
        assert 0
