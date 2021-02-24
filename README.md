DCI: Deep Cluster Infomax
===

This is the PyTorch implementation of decoupled training with DCI. We will public the implementations of baseline models later.

Requirements
---
pytorch = 1.7.1

Running examples
---
For decoupled training with DCI (GIN encoder), execute:
    
    python main_dci.py --dataset reddit --lr 0.01 --epochs 500 --num_clusters 9 to run the Reddit example.
    python main_dci.py --dataset wiki --lr 0.01 --epochs 500 --num_clusters 16 to run the Wiki example.
    python main_dci.py --dataset alpha --lr 0.01 --epochs 500 --num_clusters 45 to run the Alpha example.
    python main_dci.py --dataset amazon --lr 0.01 --epochs 500 --num_clusters 4 to run the Amazon example.


For joint training (GIN encoder), execute:

    python main_gin.py --dataset reddit --lr 0.01 --epochs 500 to run the Reddit example.
    python main_gin.py --dataset wiki --lr 0.01 --epochs 500 to run the Wiki example.
    python main_gin.py --dataset alpha --lr 0.01 --epochs 500 to run the Alpha example.
    python main_gin.py --dataset amazon --lr 0.01 --epochs 500 to run the Amazon example.
