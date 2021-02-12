import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn import metrics
import time
import random

from util import load_data_block
from models.model_test import ModelTest
from models.dci import DCI
from models.mlp import MLP

import warnings
warnings.filterwarnings("ignore", message="Numerical issues were encountered ")

sm = torch.nn.Softmax()
sig = torch.nn.Sigmoid()

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def trainFull(args, model, device, train_graph, criterion1, nb_nodes):
    model.train()

    batch_size = 1
    loc = train_graph[-1]

    output = model(train_graph[1], train_graph[2], train_graph[0], None, None, None, criterion1, loc, nb_nodes)
    loss = output
        
    return loss

def test(args, model_pretrain, device, test_graph, criterion_tune, fold_idx, feats_num, num_blocks, num_user, num_object):
    model = ModelTest(args.num_layers, args.num_mlp_layers, feats_num, args.hidden_dim, num_blocks, args.final_dropout, args.learn_eps, args.neighbor_pooling_type, num_user, num_object, device).to(device)
    
    pretrained_dict = model_pretrain.state_dict()
    model_dict = model.state_dict()

    pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    res = []
    for epoch in range(1, args.finetune_epochs+1):
        model.train()

        output = model(test_graph[1], test_graph[0])
        train_idx = test_graph[2]
        if args.dataset == 'alpha' or args.dataset == 'amazon':
            labels = torch.LongTensor(test_graph[-1]).to(device)
            loss = criterion_tune(output[labels[train_idx, 0]], torch.reshape(labels[train_idx, 1].float(), (-1, 1)))
        else:
            labels = torch.LongTensor(test_graph[-1][:, 1]).to(device)
            loss = criterion_tune(output[train_idx], torch.reshape(labels[train_idx].float(), (-1, 1)))
        #backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # testing
        model.eval()

        output = model(test_graph[1], test_graph[0])
        pred = sig(output)
        test_idx = test_graph[3]
        if args.dataset == 'alpha' or args.dataset == 'amazon':
            labels = test_graph[-1]
            pred = pred[labels[test_idx, 0]].detach().cpu().numpy()[:, 0].tolist()
            target = labels[test_idx, 1]
        else:
            labels = test_graph[-1][:, 1]
            pred = pred[test_idx].detach().cpu().numpy()[:, 0].tolist()
            target = labels[test_idx]
        
        false_positive_rate, true_positive_rate, _ = metrics.roc_curve(target, pred)
        auc = metrics.auc(false_positive_rate, true_positive_rate)

        res.append(auc)
    res = np.array(res)
    return np.max(res), res


def main():
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neurasl net')
    parser.add_argument('--dataset', type=str, default="wiki",
                        help='name of dataset (default: wiki)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='input batch size for training (default: 1024)')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--num_predictor_mlp_layers', type=int, default=1,
                        help='number of layers for MLP for the predictor (default: 1). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='number of hidden units (default: 128)')
    parser.add_argument('--finetune_epochs', type=int, default=100,
                        help='number of finetune epochs (default: 100)')
    parser.add_argument('--num_clusters', type=int, default=9,
                        help='number of finetune epochs (default: 9)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--learn_eps', action="store_true",
                                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    args = parser.parse_args()

    setup_seed(0)
    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    # Data loading
    edge_index, feats, split_idx, label, num_blocks, num_user, num_object, loc = load_data_block(args.dataset, args.num_clusters)
    idx = np.random.permutation(num_user+num_object)
    shuf_feats = feats[idx, :]

    criterion1 = nn.BCEWithLogitsLoss()
            
    model = DCI(args.num_layers, args.num_mlp_layers, feats.shape[1], args.hidden_dim, num_blocks, args.final_dropout, args.learn_eps, args.neighbor_pooling_type, num_user, num_object, device).to(device)
    
    optimizer_train = optim.Adam(model.parameters(), lr=args.lr)
            
    train_graph = (torch.tensor(edge_index), torch.FloatTensor(feats).to(device), torch.FloatTensor(shuf_feats).to(device), loc)
    ttttt = []
    for epoch in range(1, args.epochs + 1):
        loss = trainFull(args, model, device, train_graph, criterion1, num_user+num_object)
        
        ttttt.append(loss.detach().cpu().numpy().tolist())
        if epoch >= 10 and np.std(ttttt[epoch-10: epoch])<1e-4:
            break

        if optimizer_train is not None:
            optimizer_train.zero_grad()
            loss.backward()         
            optimizer_train.step()
    
    fold_idx = 1
    every_fold_auc = []
    for (train_idx, test_idx) in split_idx:
        criterion_tune = nn.BCEWithLogitsLoss()
        test_graph = (torch.tensor(edge_index), torch.FloatTensor(feats).to(device), train_idx, test_idx, label)
        tmp_auc, res = test(args, model, device, test_graph, criterion_tune, fold_idx, feats.shape[1], num_blocks, num_user, num_object)
        every_fold_auc.append(tmp_auc)
        fold_idx += 1
        print('Results of 10 folds: ', every_fold_auc)
        
    print('Average AUC over 10 folds: ', np.mean(every_fold_auc))

if __name__ == '__main__':
    main()