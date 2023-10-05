# Code of DEGT: Dynamic Effective Graph Transformer for Anomalous Node Detection on Dynamic Graphs

## Environments

Our experiments are conducted on the following evironmental settings. To ensure reproductivity, we strongly recommand that you run our code on the same settings.

- GPU: NVIDIA RTX 3090 24G
- CUDA Version: 11.3
- Pytorch Version: 1.12.0
- Python Version: 3.9.0
- DGL Version: 1.1.1

## Usage

Training: 

```
python main.py
```

All the parameters are listed in ```main.py```

We use three random seeds: 0, 1, 42. The results can be different when run multiple times with the same seed, which is caused by randomness of ```dgl.sparse``` and it is normal.

You can download datasets from link in our paper. The code of preprocessing will be published soon.
