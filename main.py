import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model import GTN
import pdb, random
import pickle
from parsing import args_parser
from utils import f1_score, adjacency, accuracy
import utils
from data.mpnnr_data import data

# from data.mpnnr_data import utils as mpnn_r_utils
device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    # sparse_mx = sp.coo_matrix(sparse_mx)
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to(device)

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
        for i, edge in enumerate(edges):  # each edge type specifies an adjacency matrix
            if i == 0:
                A = torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)
            else:
                A = torch.cat([A, torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
        A = torch.cat([A, torch.eye(num_nodes).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)  # 加入A^0=I
        A = A.to(device)
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

        with open('data/' + args.dataset + '/node_features.pkl', 'rb') as f:
            node_features = pickle.load(f)
        with open('data/' + args.dataset + '/edges.pkl', 'rb') as f:
            edges = pickle.load(f)
        with open('data/' + args.dataset + '/labels.pkl', 'rb') as f:
            labels = pickle.load(f)
        num_nodes = edges[0].shape[0]
        A = []


        for i, edge in enumerate(edges):  # each edge type specifies an adjacency matrix
            if i == 0:
                A = torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)
            else:
                A = torch.cat([A, torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
        A = torch.cat([A, torch.eye(num_nodes).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)  # 加入A^0=I
        # A = A.permute(2, 0, 1)

        node_features = torch.from_numpy(node_features).type(torch.FloatTensor)
        train_node = torch.from_numpy(np.array(labels[0])[:, 0]).type(torch.LongTensor)
        train_target = torch.from_numpy(np.array(labels[0])[:, 1]).type(torch.LongTensor)
        valid_node = torch.from_numpy(np.array(labels[1])[:, 0]).type(torch.LongTensor)
        valid_target = torch.from_numpy(np.array(labels[1])[:, 1]).type(torch.LongTensor)
        test_node = torch.from_numpy(np.array(labels[2])[:, 0]).type(torch.LongTensor)
        test_target = torch.from_numpy(np.array(labels[2])[:, 1]).type(torch.LongTensor)

        num_classes = torch.max(train_target).item() + 1

    final_f1 = 0
    for l in range(1):
        model = GTN(num_edge=A.shape[-1],
                    num_channels=num_channels,
                    w_in=node_features.shape[1],
                    w_out=node_dim,
                    num_class=num_classes,
                    num_layers=num_layers,
                    norm=norm).to(device)
        print(model)
        if adaptive_lr == 'false':
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0.0001)
        else:
            optimizer = torch.optim.Adam([{'params': model.weight},
                                          {'params': model.linear1.parameters()},
                                          {'params': model.linear2.parameters()},
                                          {"params": model.layers.parameters(), "lr": 0.1}
                                          ], lr=0.005, weight_decay=0.001)
        loss = nn.CrossEntropyLoss()
        # Train & Valid & Test
        best_val_loss = 10000
        best_test_loss = 10000
        best_train_loss = 10000
        best_train_f1 = 0
        best_val_f1 = 0
        best_test_f1 = 0
        best_test_acc = 0
        best_epoch = 0

        # torch.autograd.set_detect_anomaly(True)
        for i in range(epochs):
            for param_group in optimizer.param_groups:
                if param_group['lr'] > 0.005:
                    param_group['lr'] = param_group['lr'] * 0.9
            model.zero_grad()
            model.train()
            loss, y_train, Ws = model(A, node_features, train_node, train_target)

            loss.backward()
            nn.utils.clip_grad_norm_(list(model.parameters()), 5, norm_type=2)
            optimizer.step()

            train_acc = accuracy(torch.argmax(y_train.detach(), dim=1), train_target)
            train_f1 = torch.mean(
                f1_score(torch.argmax(y_train.detach(), dim=1), train_target, num_classes=num_classes)).cpu().numpy()
            # for n,p in model.named_parameters():
            #     print(n,p)
            #     break
            model.eval()
            # Valid
            with torch.no_grad():
                val_loss, y_valid, _ = model.forward(A, node_features, valid_node, valid_target)
                val_f1 = torch.mean(
                    f1_score(torch.argmax(y_valid, dim=1), valid_target, num_classes=num_classes)).cpu().numpy()
                test_loss, y_test, W = model.forward(A, node_features, test_node, test_target)
                test_f1 = torch.mean(
                    f1_score(torch.argmax(y_test, dim=1), test_target, num_classes=num_classes)).cpu().numpy()
                test_acc = accuracy(torch.argmax(y_test, dim=1), test_target)

            if i % 20 == 0:
                print('Epoch:  ', i + 1)
                print('Train - Loss: {}, Macro_F1: {}, acc: {}'.format(loss.detach().cpu().numpy(), train_f1,
                                                                       train_acc))
                print('Valid - Loss: {}, Macro_F1: {}'.format(val_loss.detach().cpu().numpy(), val_f1))
                print('Test - Loss: {}, Macro_F1: {} test_acc: {} \n'.format(test_loss.detach().cpu().numpy(), test_f1,
                                                                             test_acc))
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_f1 = test_f1
                best_epoch = i
            if val_f1 > best_val_f1:
                best_val_loss = val_loss.detach().cpu().numpy()
                best_test_loss = test_loss.detach().cpu().numpy()
                best_train_loss = loss.detach().cpu().numpy()
                best_train_f1 = train_f1
                best_val_f1 = val_f1

        print('---------------Best Results--------------------')
        print('Train - Loss: {}, Macro_F1: {}'.format(best_train_loss, best_train_f1))
        print('Valid - Loss: {}, Macro_F1: {}'.format(best_val_loss, best_val_f1))
        print('Test - Loss: {}, Macro_F1: {}, acc: {} at epoch: {}'.format(best_test_loss, best_test_f1, best_test_acc, best_epoch))
        final_f1 += best_test_f1
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

