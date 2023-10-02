import torch
import scipy.sparse as sp
import dgl.sparse as dglsp


def random_walk_pe(x, edge_index, k=8):
    N = x.shape[0]
    A = dglsp.spmatrix(edge_index, shape=(N, N)).coalesce().cuda()
    A = dglsp.sp_div_v(A, (A.sum(dim=1) + 1e-30))
    pe = [sp.coo_matrix((A.val.cpu(), A.indices().cpu()), shape=(N, N)).diagonal()]
    A_power = A
    for i in range(k - 1):
        print(i + 1)
        A_power = A_power @ A
        pe.append(sp.coo_matrix((A_power.val.cpu(), A_power.indices().cpu()), shape=(N, N)).diagonal())
    pe = torch.tensor(np.stack(pe, 1))
    x = torch.cat((x, pe), dim=1)
    return x


def add_label_feature(x, y):
    y = y.clone()
    y[y == 0] = 1
    y -= 1
    y_one_hot = F.one_hot(y).squeeze()
    return torch.cat((x, y_one_hot), dim=1)
