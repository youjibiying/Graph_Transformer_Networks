import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from matplotlib import pyplot as plt
import pdb


class GTN(nn.Module):
    
    def __init__(self, num_edge, num_channels, w_in, w_out, num_class,num_layers,norm):
        super(GTN, self).__init__()
        self.num_edge = num_edge
        self.num_channels = num_channels
        self.w_in = w_in
        self.w_out = w_out
        self.num_class = num_class
        self.num_layers = num_layers
        self.is_norm = norm
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GTLayer(num_edge, num_channels, first=True))
            else:
                layers.append(GTLayer(num_edge, num_channels, first=False))
        self.layers = nn.ModuleList(layers)
        self.weight = nn.Parameter(torch.Tensor(w_in, w_out))
        self.bias = nn.Parameter(torch.Tensor(w_out))
        self.loss = nn.CrossEntropyLoss()
        self.linear1 = nn.Linear(self.w_out*self.num_channels, self.w_out)
        self.linear2 = nn.Linear(self.w_out, self.num_class)
        self.dropout = nn.Dropout(p=0.5)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def gcn_conv(self,X,H):
        X = torch.mm(X, self.weight)
        H = self.norm(H, add=True) # A D^{-1}
        return torch.spmm(H.t(),X)

    def normalization(self, H):
        for i in range(self.num_channels):
            if i==0:
                H_ = self.norm(H[i,:,:]).unsqueeze(0)
            else:
                H_ = torch.cat((H_,self.norm(H[i,:,:]).unsqueeze(0)), dim=0)
        return H_

    def norm(self, H, add=False):
        H = H.t()
        # H[range(H.shape[0]), range(H.shape[0])] = 0
        H = H * ((torch.eye(H.shape[0])==0).type_as(H))

        if add == False:
            pass
            # H = H*((torch.eye(H.shape[0])==0).type(torch.FloatTensor))
        else:
            H = H + torch.eye(H.shape[0]).type_as(H)
        deg = torch.sum(H, dim=1)
        # deg_inv = deg.pow(-1)
        deg[deg == 0] = 1e9
        deg_inv = torch.pow(deg, -1)
        deg_inv[torch.isinf(deg_inv)] = 0
        # deg_inv[deg_inv == float('inf')] = 0
        deg_inv = torch.diag(deg_inv)
        # deg_inv = deg_inv*torch.eye(H.shape[0]).type(torch.FloatTensor)
        H = deg_inv @ H # D^{-1}*A
        H = H.t() # 列和为1 A*D^{-1}
        return H

    def forward(self, A, X, target_x, target):
        A = A.unsqueeze(0).permute(0,3,1,2)  # ACM : 1,5,8994,8994
        Ws = []
        for i in range(self.num_layers):
            if i == 0:
                H, W = self.layers[i](A)
            else:
                H = self.normalization(H) # 每个type adj = A*D^{-1}, 也就是列和为1
                H, W = self.layers[i](A, H)
            Ws.append(W)
        
        #H,W1 = self.layer1(A)
        #H = self.normalization(H)
        #H,W2 = self.layer2(A, H)
        #H = self.normalization(H)
        #H,W3 = self.layer3(A, H)

        ## Z=||\sigma(D^{-1}A^{l}XW)
        # H1=H
        # H = A.permute(2,0,1)
        # print(torch.norm((H1-H),1))
        for i in range(self.num_channels):
            if i==0:
                X_ = F.relu(self.dropout(self.gcn_conv(X,H[i]))) # (D+I)^{-1}*A*X*W
            else:
                X_tmp = F.relu(self.dropout(self.gcn_conv(X,H[i])))
                X_ = torch.cat((X_,X_tmp), dim=1) # ensemble

        X_ = self.linear1(X_)
        X_ = F.relu(self.dropout(X_))
        y = self.linear2(X_[target_x])
        loss = self.loss(y, target)
        return loss, y, Ws

    def forward1(self, A, X, target_x, target):
        # A = A.unsqueeze(0).permute(0, 3, 1, 2)  # ACM : 1,5,8994,8994
        # Ws = []
        # for i in range(self.num_layers):
        #     if i == 0:
        #         H, W = self.layers[i](A)
        #     else:
        #         H = self.normalization(H)  # 每个type adj = A*D^{-1}, 也就是列和为1
        #         H, W = self.layers[i](A, H)
        #     Ws.append(W)

        # H,W1 = self.layer1(A)
        # H = self.normalization(H)
        # H,W2 = self.layer2(A, H)
        # H = self.normalization(H)
        # H,W3 = self.layer3(A, H)

        ## Z=||\sigma(D^{-1}A^{l}XW)
        H = A.permute(2,0,1)
        Ws= 0
        for i in range(self.num_channels):
            if i == 0:
                X_ = F.relu(self.gcn_conv(X, H[i]))  # (D+I)^{-1}*A*X*W
            else:
                X_tmp = F.relu(self.gcn_conv(X, H[i]))
                X_ = torch.cat((X_, X_tmp), dim=1)  # ensemble

        X_ = self.linear1(X_)
        X_ = F.relu(X_)
        y = self.linear2(X_[target_x])
        loss = self.loss(y, target)
        return loss, y, Ws

class GTLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        if self.first == True:
            self.conv1 = GTConv(in_channels, out_channels) #
            self.conv2 = GTConv(in_channels, out_channels)
        else:
            self.conv1 = GTConv(in_channels, out_channels)
    
    def forward(self, A, H_=None):
        W=None
        if self.first == True: # 第一次层，需要产生两个
            a = self.conv1(A)
            b = self.conv2(A)
            H = torch.bmm(a,b)
            # H = torch.bmm(a,b)
            # W = [(F.softmax(self.conv1.weight, dim=1)).detach(),(F.softmax(self.conv2.weight, dim=1)).detach()]
        else:
            a = self.conv1(A) # 生成新的Q^(l)
            H = torch.bmm(H_,a) # A^{l-1}*Q^{l}
            # W = [(F.softmax(self.conv1.weight, dim=1)).detach()]
        return H,W

class GTConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels,1,1), requires_grad=True)
        self.bias = None
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.reset_parameters()
    def reset_parameters(self):
        n = self.in_channels
        nn.init.constant_(self.weight, 0.1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A):
        # if (A == float('-inf')).sum() or (A == float('inf')).sum():
        #     print('-=----inf')
        # A = torch.sum(A.contiguous()*F.softmax(self.weight, dim=1), dim=1) # self.weight.shape = [2,5,1,1]
        A = torch.sum(A.contiguous()*F.softmax(self.weight, dim=1), dim=1) # self.weight.shape = [2,5,1,1]
        return A
