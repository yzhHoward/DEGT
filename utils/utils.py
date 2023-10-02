import torch
import numpy as np
import os
import random
import sys
import logging
import dgl
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support


def check_input(y_true, y_pred):
    '''
        y_true: numpy ndarray or torch tensor of shape (num_node)
        y_pred: numpy ndarray or torch tensor of shape (num_node, num_tasks)
    '''

    # converting to torch.Tensor to numpy on cpu
    if torch is not None and isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if torch is not None and isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    ## check type
    if not (isinstance(y_true, np.ndarray) and isinstance(y_true, np.ndarray)):
        raise RuntimeError('Arguments to Evaluator need to be either numpy ndarray or torch tensor')

    if not y_pred.ndim == 2:
        raise RuntimeError('y_pred must to 2-dim arrray, {}-dim array given'.format(y_true.ndim))

    return y_true, y_pred


def evaluate(y_true, y_pred, all=False):
    y_true, y_pred = check_input(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_pred[:, 1])

    if not all:
        return auroc
    else:
        auprc = average_precision_score(y_true, y_pred[:, 1], average='weighted')
        best_f1 = 0
        for thres in np.linspace(0.05, 0.95, 19):
            preds = np.zeros_like(y_true)
            preds[y_pred[:, 1] > thres] = 1
            precision, recall, mf1, _ = precision_recall_fscore_support(y_true, preds, average='macro')
            if mf1 > best_f1:
                best_f1 = mf1
                gmean = (precision * recall) ** 0.5
        acc = float(np.sum(y_true == y_pred.argmax(axis=-1))) / len(y_true)
        return {
            'auroc': auroc,
            'auprc': auprc,
            'f1': best_f1,
            'gmean': gmean,
            'acc': acc
        }


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)


def prepare_folder(name, model_name):
    model_dir = f'./model_files/{name}/{model_name}/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir


def init_logging(log_root, models_root=None):
    log_root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s")
    if models_root is not None:
        handler_file = logging.FileHandler(
            os.path.join(models_root, "training.log"))
        handler_file.setFormatter(formatter)
        log_root.addHandler(handler_file)
    handler_stream = logging.StreamHandler(sys.stdout)
    handler_stream.setFormatter(formatter)
    log_root.addHandler(handler_stream)
    