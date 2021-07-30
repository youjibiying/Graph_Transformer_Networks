import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model_sparse import GTN
from matplotlib import pyplot as plt
import pdb
from torch_geometric.utils import dense_to_sparse, f1_score, accuracy
from torch_geometric.data import Data
import torch_sparse
import pickle
#from mem import mem_report
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import argparse
import random
from  utils import  adjacency, accuracy
from parsing import args_parser
from data.mpnnr_data import data
import utils



device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')

def main(args):
    epochs = args.epoch
    node_dim = args.node_dim
    num_channels = args.num_channels
    lr = args.lr
    weight_decay = args.weight_decay
    num_layers = args.num_layers
    norm = args.norm
    adaptive_lr = args.adaptive_lr

    if args.type == 'hgnn':
        dataset, splits = data.load(args)
        edges = adjacency(dataset['hypergraph'], args=args)
        num_nodes = edges[0].shape[0]
        # A = [sparse_mx_to_torch_sparse_tensor(H) for H in edges]
        # A = torch.stack(A,dim=0)
        A = []
        for i, edge in enumerate(edges):  # each edge type specifies an adjacency matrix
            edge_tmp = torch.from_numpy(np.vstack((edge.row,edge.col))).type(torch.cuda.LongTensor)
            value_tmp = torch.from_numpy(edge.data).type(torch.cuda.FloatTensor)
            A.append((edge_tmp,value_tmp))
        edge_tmp = torch.stack((torch.arange(0, num_nodes), torch.arange(0, num_nodes))).type(torch.cuda.LongTensor)
        value_tmp = torch.ones(num_nodes).type(torch.cuda.FloatTensor)
        A.append((edge_tmp, value_tmp))


        node_features = torch.tensor(utils.normalise(dataset['X'])).type(torch.FloatTensor).to(device)
        labels = np.where(dataset['Y'])[1]
        train_node = torch.tensor(splits['train']).type(torch.LongTensor).to(device)
        train_target = torch.tensor(labels[splits['train']]).type(torch.LongTensor).to(device)
        valid_node = torch.tensor(splits['test']).type(torch.LongTensor).to(device)
        valid_target = torch.tensor(labels[splits['test']]).type(torch.LongTensor).to(device)
        test_node = torch.tensor(splits['test']).type(torch.LongTensor).to(device)
        test_target = torch.tensor(labels[splits['test']]).type(torch.LongTensor).to(device)
        num_classes = dataset['c']  # torch.tensor(dataset['c']).type(torch.FloatTensor)
    else:

        with open('data/'+args.dataset+'/node_features.pkl','rb') as f:
            node_features = pickle.load(f)
        with open('data/'+args.dataset+'/edges.pkl','rb') as f:
            edges = pickle.load(f)
        with open('data/'+args.dataset+'/labels.pkl','rb') as f:
            labels = pickle.load(f)


        num_nodes = edges[0].shape[0]
        A = []

        for i,edge in enumerate(edges):
            # 取出非0元所在的坐标
            edge_tmp = torch.from_numpy(np.vstack((edge.nonzero()[0], edge.nonzero()[1]))).type(torch.cuda.LongTensor)
            # 取出对应的values
            value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.cuda.FloatTensor)
            A.append((edge_tmp,value_tmp))
        edge_tmp = torch.stack((torch.arange(0,num_nodes),torch.arange(0,num_nodes))).type(torch.cuda.LongTensor)
        value_tmp = torch.ones(num_nodes).type(torch.cuda.FloatTensor)
        A.append((edge_tmp,value_tmp))

        node_features = torch.from_numpy(node_features).type(torch.cuda.FloatTensor)
        train_node = torch.from_numpy(np.array(labels[0])[:,0]).type(torch.cuda.LongTensor)
        train_target = torch.from_numpy(np.array(labels[0])[:,1]).type(torch.cuda.LongTensor)

        valid_node = torch.from_numpy(np.array(labels[1])[:,0]).type(torch.cuda.LongTensor)
        valid_target = torch.from_numpy(np.array(labels[1])[:,1]).type(torch.cuda.LongTensor)
        test_node = torch.from_numpy(np.array(labels[2])[:,0]).type(torch.cuda.LongTensor)
        test_target = torch.from_numpy(np.array(labels[2])[:,1]).type(torch.cuda.LongTensor)


        num_classes = torch.max(train_target).item()+1

    train_losses = []
    train_f1s = []
    val_losses = []
    test_losses = []
    val_f1s = []
    test_f1s = []
    final_f1 = 0
    for cnt in range(1):
        best_val_loss = 10000
        best_test_loss = 10000
        best_train_loss = 10000
        best_train_f1 = 0
        best_val_f1 = 0
        best_test_f1 = 0
        best_test_acc = 0

        model = GTN(num_edge=len(A),
                        num_channels=num_channels,
                        w_in = node_features.shape[1],
                        w_out = node_dim,
                        num_class=num_classes,
                        num_nodes = node_features.shape[0],
                        num_layers= num_layers)
        model.cuda()
        if adaptive_lr == 'false':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam([{'params':model.gcn.parameters()},
                                        {'params':model.linear1.parameters()},
                                        {'params':model.linear2.parameters()},
                                        {"params":model.layers.parameters(), "lr":0.5}
                                        ], lr=0.005, weight_decay=0.001)
        loss = nn.CrossEntropyLoss()
        Ws = []
        for i in range(args.epoch):
            for param_group in optimizer.param_groups:
                if param_group['lr'] > 0.005:
                    param_group['lr'] = param_group['lr'] * 0.9
            model.train()
            model.zero_grad()
            loss, y_train, _ = model(A, node_features, train_node, train_target)
            loss.backward()
            optimizer.step()
            train_f1 = torch.mean(f1_score(torch.argmax(y_train,dim=1), train_target, num_classes=3)).cpu().numpy()
            train_acc = accuracy(torch.argmax(y_train.detach(), dim=1), train_target)
            model.eval()
            # Valid
            with torch.no_grad():
                val_loss, y_valid,_ = model.forward(A, node_features, valid_node, valid_target)
                val_f1 = torch.mean(f1_score(torch.argmax(y_valid,dim=1), valid_target, num_classes=3)).cpu().numpy()
                # print('Valid - Loss: {}, Macro_F1: {}'.format(val_loss.detach().cpu().numpy(), val_f1))
                test_loss, y_test,W = model.forward(A, node_features, test_node, test_target)
                test_f1 = torch.mean(f1_score(torch.argmax(y_test,dim=1), test_target, num_classes=3)).cpu().numpy()
                test_acc = accuracy(torch.argmax(y_test,dim=1), test_target)
                # print('Test - Loss: {}, Macro_F1: {}, Acc: {}\n'.format(test_loss.detach().cpu().numpy(), test_f1, test_acc))
                if val_f1 > best_val_f1:
                    best_val_loss = val_loss.detach().cpu().numpy()
                    best_test_loss = test_loss.detach().cpu().numpy()
                    best_train_loss = loss.detach().cpu().numpy()
                    best_train_f1 = train_f1
                    best_val_f1 = val_f1
                    best_test_f1 = test_f1

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_f1 = test_f1
                best_epoch = i
            if i % 20 == 0:
                print('Epoch:  ', i + 1)
                print('Train - Loss: {:5f}, Macro_F1: {:4f}, acc: {:4f}'.format(loss.detach().cpu().numpy(), train_f1,
                                                                       train_acc))
                print('Valid - Loss: {:5f}, Macro_F1: {:4f}'.format(val_loss.detach().cpu().numpy(), val_f1))
                print('Test - Loss: {:5f}, Macro_F1: {:4f} test_acc: {:4f} best_epoch: {}\n'.format(test_loss.detach().cpu().numpy(), test_f1,
                                                                             test_acc, best_epoch))
            torch.cuda.empty_cache()

        print('---------------Best Results--------------------')
        print('Train - Loss: {:5f}, Macro_F1: {:4f}'.format(best_test_loss, best_train_f1))
        print('Valid - Loss: {:5f}, Macro_F1: {:4f}'.format(best_val_loss, best_val_f1))
        print('Test - Loss: {:5f}, Macro_F1: {:4f}, acc: {:4f} at epoch: {}'.format(best_test_loss, best_test_f1, best_test_acc,
                                                                           best_epoch))
    return best_test_acc

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    ## cuda 问题
    torch.cuda.current_device()
    torch.cuda._initialized = True


if __name__ == '__main__':
    # todo: set random seed
    setup_seed(1000)
    args = args_parser()
    results = []
    for i in range(1, 11):
        print('split:', i)
        args.split = i
        test_acc = main(args)
        results.append(test_acc)
        print('Acc_array:', results)
    results = np.array(results)
    print(f"avg_test_acc={results.mean()} \n"
          f"std={results.std()}")