import torch
import torch.nn as nn
from layers import GraphCNN, AvgReadout, Discriminator
import torch.nn.functional as F
import math
import sys
sys.path.append("models/")
import numpy as np

class DCI(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, final_dropout, learn_eps, neighbor_pooling_type, num_user, num_object, device):
        super(DCI, self).__init__()
        self.gin = GraphCNN(num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, final_dropout, learn_eps, neighbor_pooling_type, num_user, num_object, device)
        self.read = AvgReadout()
        self.device = device
        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(hidden_dim)

    def forward(self, seq1, seq2, adj, msk, samp_bias1, samp_bias2, criterion1, loc, nb_nodes):
        h_1 = torch.unsqueeze(self.gin(adj, seq1), 0)
        h_2 = torch.unsqueeze(self.gin(adj, seq2), 0)

        loss = 0
        batch_size = 1
        for i in range(loc.shape[1]):
            node_idx = np.where(loc[:, i])[0]
            h_1_block = torch.unsqueeze(torch.squeeze(h_1)[node_idx], 0)
            c_block = self.read(h_1_block, msk)
            c_block = self.sigm(c_block)
            h_2_block = torch.unsqueeze(torch.squeeze(h_2)[node_idx], 0)
            lbl_1 = torch.ones(batch_size, len(node_idx))
            lbl_2 = torch.zeros(batch_size, len(node_idx))
            lbl = torch.cat((lbl_1, lbl_2), 1).to(self.device)

            ret = self.disc(c_block, h_1_block, h_2_block, samp_bias1, samp_bias2)
        
            loss += criterion1(ret, lbl)

        return loss / loc.shape[1]