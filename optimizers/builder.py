from torch.optim import Adam, AdamW, SGD, RMSprop


def choose_optimizer(model, config):
    """
    Choose metric function under ./optimizers and standard optimizers
    Args:
        model (torch.nn.Module): Model
        config (dict): Config
            Required keys: optimizer, optimizer -> type
            Optional keys: optimizer -> params (which contains key-value pairs accepted by optimizer class)
    Returns:
        optimizer (torch.optim.Optimizer): Optimizer
    """
    if config['optimizer']['type'] == 'adam':
        return Adam(params=filter(lambda p: p.requires_grad, model.parameters()), **config['optimizer']['params'])
    elif config['optimizer']['type'] == 'adamw':
        return AdamW(params=filter(lambda p: p.requires_grad, model.parameters()), **config['optimizer']['params'])
    elif config['optimizer']['type'] == 'sgd':
        return SGD(params=filter(lambda p: p.requires_grad, model.parameters()), **config['optimizer']['params'])
    elif config['optimizer']['type'] == 'rmsprop':
        return RMSprop(params=filter(lambda p: p.requires_grad, model.parameters()), **config['optimizer']['params'])
    else:
        raise NotImplementedError(f"Unknown optimizer type: {config['optimizer']['type']}")
