import torch 

class Conf:
    def __init__(self, args) -> None:
        conf = args
        if conf.device == 'default': 
            conf.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif conf.device in ['cuda', 'cpu']:
            pass 
        else:
            raise ValueError("Device is not appliable.")