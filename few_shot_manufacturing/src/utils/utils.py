import random
import numpy as np
import torch
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1-decay_rate)**epoch)
    print(" Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def print_params(model):
    print('total parameters:', sum([np.prod(list(p.size())) for p in model.parameters() if p.requires_grad]))


def regression_metrics(gold_results, pred_results):
    res_rmse = np.sqrt(mean_squared_error(gold_results, pred_results))
    res_mse = mean_squared_error(gold_results, pred_results)
    res_mae = mean_absolute_error(gold_results, pred_results)
    return res_rmse, res_mse, res_mae
