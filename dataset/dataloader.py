import dgl
import torch
from torch.utils.data import Dataset, DataLoader


class RawDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class KHopDataLoaderDGL(DataLoader):
    def __init__(self, dataset, graph, k, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        self.graph = graph
        self.collate_fn = self.collate_merge_subgraph
        self.k = k

    def collate_merge_subgraph(self, batch_index):
        subgraph, inverse_indices = dgl.khop_in_subgraph(self.graph, batch_index, self.k)
        
        return torch.isin(subgraph.nodes(), inverse_indices), subgraph.ndata['feature'], \
            subgraph.ndata['label'], torch.stack(subgraph.edges()), \
            subgraph.edata['feature'] if 'feature' in subgraph.edata else None, \
            subgraph.edata['timestamp'], subgraph.edata['direct'] if 'direct' in subgraph.edata else None
    
