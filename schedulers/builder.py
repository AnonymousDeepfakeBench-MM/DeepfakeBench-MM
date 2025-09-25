from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR


def choose_scheduler(optimizer, config):
    """
    Args:
        optimizer (torch.optim.Optimizer): Optimizer
        config (dict): Config
            Optional keys: scheduler, scheduler -> type, scheduler -> params (which contains key-value pairs accepted by scheduler class)
    Returns:
        scheduler
    """
    if config.get('scheduler', None) is None:
        return None
    elif config['scheduler']['type'] == 'step':
        return StepLR(optimizer, **config['scheduler']['params'])
    elif config['scheduler']['type'] == 'cosine':
        return CosineAnnealingLR(optimizer, **config['scheduler']['params'])
    else:
        raise NotImplementedError(f"Unknown scheduler type: {config['scheduler']['type']}")
