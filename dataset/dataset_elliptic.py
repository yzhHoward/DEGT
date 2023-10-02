import pickle
import torch
import dgl

from utils.feat_func import random_walk_pe, add_label_feature


def to_undirected(edge_index, edge_timestamp):
    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)

    edge_timestamp = torch.cat([edge_timestamp, edge_timestamp], dim=0)
    return edge_index, edge_timestamp


def data_process(x, y, edge_index, edge_timestamp, train_mask, valid_mask, test_mask):
    x = add_label_feature(x, y)
    x = random_walk_pe(x, edge_index)
    edge_index, edge_timestamp = to_undirected(edge_index, edge_timestamp)
        
    data = dgl.graph((edge_index[0], edge_index[1]), num_nodes=x.shape[0])
    data.ndata['feature'] = x
    data.ndata['label'] = y
    data.edata['timestamp'] = edge_timestamp
    data.train_mask = train_mask
    data.valid_mask = valid_mask
    data.test_mask = test_mask

    edge_direct = torch.ones(edge_timestamp.size(0), dtype=torch.long)
    edge_direct[: edge_timestamp.size(0) // 2] = 0
    data.edata['direct'] = edge_direct
    return data


def read_elliptic():
    print("reading elliptic...")
    data = pickle.load(open('data/elliptic_bitcoin_dataset/elliptic.pkl', 'rb'))
    x = data['node_feature']
    y = data['node_label']

    edge_index = data['edge_index']
    edge_timestamp = data['edge_timestamp']
    train_mask = data['train_mask']
    valid_mask = data['valid_mask']
    test_mask = data['test_mask']
    
    x = torch.tensor(x, dtype=torch.float).contiguous()
    x = (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True)
    y = torch.tensor(y, dtype=torch.long)
    edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()
    edge_timestamp = torch.tensor(edge_timestamp, dtype=torch.float) - 1
    train_mask = torch.tensor(train_mask, dtype=torch.long)
    valid_mask = torch.tensor(valid_mask, dtype=torch.long)
    test_mask = torch.tensor(test_mask, dtype=torch.long)

    data = data_process(x, y, edge_index, edge_timestamp, train_mask, valid_mask, test_mask)
    return data
