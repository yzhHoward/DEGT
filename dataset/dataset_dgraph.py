import numpy as np
import torch
import dgl

from utils.feat_func import random_walk_pe, add_label_feature


def to_undirected(edge_index, edge_attr, edge_timestamp):
    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)

    edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    edge_timestamp = torch.cat([edge_timestamp, edge_timestamp], dim=0)
    return edge_index, edge_attr, edge_timestamp


def data_process(x, y, edge_index, edge_attr, edge_timestamp, train_mask, valid_mask, test_mask):
    x = add_label_feature(x, y)
    x = random_walk_pe(x, edge_index)
    edge_index, edge_attr, edge_timestamp = to_undirected(
        edge_index, edge_attr, edge_timestamp
    )

    data = dgl.graph((edge_index[0], edge_index[1]), num_nodes=x.shape[0])
    data.ndata['feature'] = x
    data.ndata['label'] = y
    data.edata['feature'] = edge_attr
    data.edata['timestamp'] = edge_timestamp

    data.train_mask = train_mask
    data.valid_mask = valid_mask
    data.test_mask = test_mask

    edge_direct = torch.ones(edge_attr.size(0), dtype=torch.long)
    edge_direct[: edge_attr.size(0) // 2] = 0
    data.edata['direct'] = edge_direct
    
    return data


def read_dgraphfin():
    print("read_dgraphfin...")
    items = np.load('data/DGraphFin/dgraphfin.npz')
    x = items["x"]
    y = items["y"]

    edge_index = items["edge_index"].transpose()
    edge_type = items["edge_type"]
    edge_timestamp = items["edge_timestamp"]
    train_mask = items["train_mask"]
    valid_mask = items["valid_mask"]
    test_mask = items["test_mask"]

    x = torch.tensor(x, dtype=torch.float).contiguous()
    x = (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True)
    y = torch.tensor(y, dtype=torch.long)
    edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()
    edge_timestamp = torch.tensor(edge_timestamp, dtype=torch.float) - 1
    edge_type = torch.tensor(edge_type, dtype=torch.long) - 1
    train_mask = torch.tensor(train_mask, dtype=torch.long)
    valid_mask = torch.tensor(valid_mask, dtype=torch.long)
    test_mask = torch.tensor(test_mask, dtype=torch.long)

    data = data_process(x, y, edge_index, edge_type, edge_timestamp, train_mask, valid_mask, test_mask)
    return data
