import argparse
import torch
import torch.nn as nn
import time
import logging

from models.model import DEGT
from dataset.dataset_dgraph import read_dgraphfin
from dataset.dataset_elliptic import read_elliptic
from dataset.dataloader import RawDataset, KHopDataLoaderDGL
from utils.utils import prepare_folder, set_seed, evaluate, init_logging


def train(model, data, optimizer, criterion):
    data = (_.cuda() if _ is not None else None for _ in data)
    index, x, y, edge_index, edge_attr, edge_timestamp, edge_direct = data
    optimizer.zero_grad()
    out, h = model(x, edge_index, edge_attr, edge_timestamp, edge_direct, return_h=True)
    loss = criterion(out[index], y[index])
    loss.backward()
    optimizer.step()
    return loss.item() * torch.sum(index).item()


@torch.no_grad()
def test(model, data, return_h=False):
    data = (_.cuda() if _ is not None else None for _ in data)
    index, x, y, edge_index, edge_attr, edge_timestamp, edge_direct = data
    if return_h:
        out, h = model(x, edge_index, edge_attr, edge_timestamp, edge_direct, return_h)
        return out[index].softmax(dim=-1), y[index], h[index].cpu()
    out = model(x, edge_index, edge_attr, edge_timestamp, edge_direct)
    return out[index].softmax(dim=-1), y[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="elliptic", choices=['DGraphFin', 'elliptic'])
    parser.add_argument("--model", type=str, default="DEGT")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hiddens", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    model_dir = prepare_folder(args.dataset, args.model)
    init_logging(logging.getLogger(), model_dir)
    logging.info(args)

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logging.info("model_dir: " + model_dir)
    set_seed(args.seed)

    if args.dataset == 'DGraphFin':
        data = read_dgraphfin()
        nlabels = 2
        args.epochs = 25
    elif args.dataset == 'elliptic':
        data = read_elliptic()
        nlabels = 2
        args.epochs = 100

    data = data.to(device)

    train_dataset = RawDataset(data.train_mask)
    valid_dataset = RawDataset(data.valid_mask)
    test_dataset = RawDataset(data.test_mask)
    train_dataloader = KHopDataLoaderDGL(train_dataset, data, args.layers, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = KHopDataLoaderDGL(valid_dataset, data, args.layers, batch_size=args.batch_size)
    test_dataloader = KHopDataLoaderDGL(test_dataset, data, args.layers, batch_size=args.batch_size)

    model = DEGT(
        in_channels=data.ndata['feature'].shape[-1],
        edge_channels=data.edata['feature'].shape[-1] if 'feature' in data.edata else None,
        hidden_channels=args.hiddens,
        out_channels=nlabels,
        num_layers=args.layers,
        num_heads=args.heads,
        dropout=args.dropout
    ).to(device)

    logging.info(f"Model {args.model} initialized")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        cur_time = time.time()
        loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader, 1):
            loss += train(model, batch, optimizer, criterion, args.loss_ratio)
            if step % 10 == 0:
                logging.info(f"Epoch {epoch:02d}, Step {step:02d}, Loss: {loss / step / args.batch_size:.4f}")
        loss /= len(data.train_mask)

        model.eval()
        pred_ys, true_ys = [], []
        for batch in valid_dataloader:
            pred_y, true_y = test(model, batch)
            pred_ys.append(pred_y)
            true_ys.append(true_y)
        pred_ys = torch.cat(pred_ys)
        true_ys = torch.cat(true_ys)
        valid_auc = evaluate(true_ys, pred_ys)

        if valid_auc >= best_auc:
            best_auc = valid_auc
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                # 'optimizer': optimizer.state_dict()
            }, model_dir + 'model.bin')
        logging.info(
            f"Epoch: {epoch:02d}, "
            f"Loss: {loss:.4f}, "
            f"Valid: {valid_auc:.2%}, "
            f"Best: {best_auc:.4%}, "
            f"Time: {time.time() - cur_time:.2f}s"
        )

    params = torch.load(model_dir + 'model.bin')
    logging.info(f"Loading best model at epoch: {params['epoch']:02d}")
    model.load_state_dict(params['model'])
    # optimizer.load_state_dict(params['optimizer'])
    model.eval()
    pred_ys, true_ys, hs = [], [], []
    for batch in test_dataloader:
        pred_y, true_y, h = test(model, batch, True)
        pred_ys.append(pred_y)
        true_ys.append(true_y)
        hs.append(h)
    pred_ys = torch.cat(pred_ys)
    true_ys = torch.cat(true_ys)
    hs = torch.cat(hs)
    test_result = evaluate(true_ys, pred_ys, all=True)
    print(
        f"Test auroc: {test_result['auroc']:.4%}, "
        f"auprc: {test_result['auprc']:.4%}, "
        f"f1: {test_result['f1']:.4%}, "
        f"gmean: {test_result['gmean']:.4%}, "
        f"acc: {test_result['acc']:.4%}"
    )

if __name__ == "__main__":
    main()
