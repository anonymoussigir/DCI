import numpy as np
import random
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from collections import Counter

random.seed(0)
np.random.seed(0)

def load_data_block(datasets, cluster_num):
    adj = np.loadtxt('./data/'+datasets+'.txt')
    adj = adj[:, 0: 2]
    num_user = len(set(adj[:, 0]))
    num_object = len(set(adj[:, 1]))
    adj[:, 1] += (np.max(adj[:, 0])+1)
    adj = adj.astype('int')
    edge_index = adj.T
    print('Load the edge_index down!')

    # load the user label
    if datasets == 'alpha' or datasets == 'amazon':
        label = np.load('./data/'+datasets+'_label.npy')
        print('Load user label down!')
        print('The spammer ratio: ', np.sum(label[:, 1]) / len(label))
        y = label
        
    else:
        label = np.load('./data/'+datasets+'_label.npy')
        y = np.zeros((len(label), 2))
        for idx in range(y.shape[0]):
            y[idx, int(label[idx])] = 1
        print('The spammer ratio: ', np.sum(label) / len(label))

    # split the train_set and validation_set
    split_idx = []
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for (train_idx, test_idx) in skf.split(y[:, 1], y[:, 1]):
        split_idx.append((train_idx, test_idx))

    # initialize the node features
    feats_lap = np.load('./feature/'+datasets+'_feature64.npy')
    feats = feats_lap

    kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(feats)
    ss_label = kmeans.labels_
    loc = np.zeros((feats.shape[0], cluster_num))

    for i in range(feats.shape[0]):
        loc[i, ss_label[i]] = 1

    num_blocks = loc.shape[1]
    
    return edge_index, feats, split_idx, y, num_blocks, num_user, num_object, loc
