from __future__ import division

import torch


def accuracy(pred, target):
    r"""Computes the accuracy of correct predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.

    :rtype: int
    """
    return (pred == target).sum().item() / target.numel()



def true_positive(pred, target, num_classes):
    r"""Computes the number of true positive predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    out = []
    for i in range(num_classes):
        out.append(((pred == i) & (target == i)).sum())

    return torch.tensor(out)



def true_negative(pred, target, num_classes):
    r"""Computes the number of true negative predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    out = []
    for i in range(num_classes):
        out.append(((pred != i) & (target != i)).sum())

    return torch.tensor(out)



def false_positive(pred, target, num_classes):
    r"""Computes the number of false positive predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    out = []
    for i in range(num_classes):
        out.append(((pred == i) & (target != i)).sum())

    return torch.tensor(out)



def false_negative(pred, target, num_classes):
    r"""Computes the number of false negative predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    out = []
    for i in range(num_classes):
        out.append(((pred != i) & (target == i)).sum())

    return torch.tensor(out)



def precision(pred, target, num_classes):
    r"""Computes the precision:
    :math:`\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}`.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`Tensor`
    """
    tp = true_positive(pred, target, num_classes).to(torch.float)
    fp = false_positive(pred, target, num_classes).to(torch.float)

    out = tp / (tp + fp)
    out[torch.isnan(out)] = 0

    return out



def recall(pred, target, num_classes):
    r"""Computes the recall:
    :math:`\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}`.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`Tensor`
    """
    tp = true_positive(pred, target, num_classes).to(torch.float)
    fn = false_negative(pred, target, num_classes).to(torch.float)

    out = tp / (tp + fn)
    out[torch.isnan(out)] = 0

    return out



def f1_score(pred, target, num_classes):
    r"""Computes the :math:`F_1` score:
    :math:`2 \cdot \frac{\mathrm{precision} \cdot \mathrm{recall}}
    {\mathrm{precision}+\mathrm{recall}}`.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`Tensor`
    """
    prec = precision(pred, target, num_classes)
    rec = recall(pred, target, num_classes)

    score = 2 * (prec * rec) / (prec + rec)
    score[torch.isnan(score)] = 0

    return score


## for mpnnr
import torch, numpy as np, scipy.sparse as sp
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import time, torch, os
from scipy.sparse import coo_matrix
import time

def adjacency_o(H):
    """
    construct adjacency for recursive hypergraph
    arguments:
    H: recursive hypergraph
    """
    A = np.eye(H['n'])
    E = H['D0']

    for k in tqdm(E):
        e = list(E[k])
        for u in e:
            A[k][u], A[u][k] = 1, 1
            for v in e:  # every node in e connect
                if u != v:  # A[v,v]已经连接好的
                    A[u][v], A[v][u] = 1, 1

    E = H['D1']
    for k in tqdm(E):
        e = list(E[k])
        for u in e:
            for v in e:
                if u != v:
                    A[u][v], A[v][u] = 1, 1  # depth-1中的边对应顶点的两两都会连上。

    return ssm2tst(symnormalise(sp.csr_matrix(A)))


def adjacency(H, args=None):
    """
    construct adjacency for recursive hypergraph
    arguments:
    H: recursive hypergraph
    """
    if args.type == 'mpnn':
        A = np.eye(H['n'])
        E = H['D0']

        for k in tqdm(E):
            e = list(E[k])
            for u in e:
                A[k][u], A[u][k] = 1, 1
                for v in e:  # every node in e connect
                    if u != v:  # A[v,v]已经连接好的
                        A[u][v], A[v][u] = 1, 1

        E = H['D1']
        for k in tqdm(E):
            e = list(E[k])
            for u in e:
                for v in e:
                    if u != v:
                        A[u][v], A[v][u] = 1, 1  # depth-1中的边对应顶点的两两都会连上。
        return ssm2tst(symnormalise(sp.csr_matrix(A)))
    else:
        # H1 = np.zeros([H['n'], max(H['D0'].keys()) + 1])
        col,row,value = [],[],[]
        for e, v in H['D0'].items():
            # print(e, v)
            # todo col row data
            row.extend(list(v)+[e])
            col.extend([e]*len(v)+[e])
            value.extend([1]*len(v)+[1])
            # H1[list(v), e] = 1
            # H1[e, e] = 1
        H1=coo_matrix((value,(row, col)),shape=(H['n'], max(H['D0'].keys()) + 1))
        # H2 = np.zeros([H['n'], len(H['D1']) + 1])
        col,row,value = [],[],[]
        for i, (e, v) in enumerate(H['D1'].items()):
            # print(e, v)
            row.extend(list(v))
            col.extend([i]*len(v))
            value.extend([1]*len(v))
            # H2[list(v), i] = 1
        H2=coo_matrix((value,(row, col)),shape=(H['n'], len(H['D1'])+1))
        A0= sp.eye(H['n'])
        t=time.time()
        A1 = _generate_G_from_H_sparse(H1, args=args,sigma=-1)
        t1 = time.time()
        A2 = _generate_G_from_H_sparse(H2, args=args,sigma=-1)
        # A3 = _generate_G_from_H_sparse(H1, args=args, sigma=-0.5)
        print( t1-t, time.time()-t1)
        return [coo_matrix(A1),coo_matrix(A2)]
        # A3 = _generate_G_from_H_sparse(H1, args=args, sigma=-0.5)
        # return [sp.csr_matrix(A1),sp.csr_matrix(A2),sp.csr_matrix(A3)] # 0.772
        # return [sp.csr_matrix(A0), sp.csr_matrix(A1),sp.csr_matrix(A2)]#,sp.csr_matrix(A3)]


def _generate_G_from_H_sparse(H, variable_weight=False, args=None, sigma = None):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if sigma is None:
        sigma = args.sigma
    print('sigma=', sigma)
    # H1 = np.array(H)
    H = coo_matrix(H)
    n_edge = H.shape[1]  # 4024
    # the weight of the hyperedge
    W = np.ones(n_edge)  # 使用权重为1
    # the degree of the node
    # DV = np.sum(H * W, axis=1)  # [2012]数组*是对应位置相乘，矩阵则是矩阵乘法 https://blog.csdn.net/TeFuirnever/article/details/88915383
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)  # [4024]
    DE = DE.tolist()[0]
    # exp_H = np.exp(H)
    # H=exp_H*H
    # H = np.mat(H)
    if args.add_self_loop:
        invDE = np.power(DE, sigma)
    else:
        invDE = np.power(DE-np.ones(n_edge), sigma)
    invDE[np.isinf(invDE)] = 0
    invDE = coo_matrix((invDE, (range(n_edge), range(n_edge))), shape=(n_edge, n_edge))
    K = H* invDE * H.T

    if args.add_self_loop:
        K+=sp.eye(H.shape[0])
    else:
        K = K.multiply(sp.eye(H.shape[0]) == 0)
        K+=sp.eye(H.shape[0])

    # if args.add_self_loop:  # lazy random  walk
    #     invDE = np.mat(np.diag(np.power(DE, sigma)))
    #
    #     # DE = np.sin(np.pi/2*(DE-DE.min())/(DE.max()-DE.min())) # pubmed 74.23 cora: 68.9
    #     # invDE = np.mat(np.diag(DE))
    #
    #     invDE[np.isinf(invDE)] = 0  # D_e ^-1
    #     T = H * (invDE) * H.T
    #     T += np.mat(np.identity(H.shape[0]))
    # else:  # HGNN
    #     # invDE = np.mat(np.diag(np.power(DE, sigma)))
    #     invDE = np.mat(np.diag(np.power(DE - np.ones(n_edge), sigma)))
    #     invDE[np.isinf(invDE)] = 0  # D_e ^-1
    #     T = H * (invDE) * H.T
    #     np.fill_diagonal(T, 1)  # 修改其适应数据集

    return K
    # DE = np.mat(np.power(np.diag(nDE)))
    # DE = np.mat(np.power(np.diag(DE)-np.diag(W),sigma))
    # T[range(H.shape[0]),range(H.shape[0])]=0
    DV = np.sum(K,0).tolist()[0]  # [2012]数组*是对应位置相乘，矩阵则是矩阵乘法 https://blog.csdn.net/TeFuirnever/article/details/88915383
    invDV = np.power(DV, -0.5)
    invDV[np.isinf(invDV)] = 0
    DV2 = coo_matrix((invDV, (range(H.shape[0]), range(H.shape[0]))), shape=(H.shape[0], H.shape[0]))

    # DV2 = np.mat(np.diag(invDV))  # D_v^-1/2
    G = DV2 * K * DV2
    if 0:
        eign_decomposition(G, plot=True, args=args)
    return G

def _generate_G_from_H(H, variable_weight=False, args=None, sigma=None):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if sigma is None:
        sigma = args.sigma
    print('sigma=', sigma)
    H = np.array(H)
    n_edge = H.shape[1]  # 4024
    # the weight of the hyperedge
    W = np.ones(n_edge)  # 使用权重为1
    # the degree of the node
    # DV = np.sum(H * W, axis=1)  # [2012]数组*是对应位置相乘，矩阵则是矩阵乘法 https://blog.csdn.net/TeFuirnever/article/details/88915383
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)  # [4024]
    # exp_H = np.exp(H)
    # H=exp_H*H
    H = np.mat(H)

    if args.add_self_loop:  # lazy random  walk
        invDE = np.mat(np.diag(np.power(DE, sigma)))

        # DE = np.sin(np.pi/2*(DE-DE.min())/(DE.max()-DE.min())) # pubmed 74.23 cora: 68.9
        # invDE = np.mat(np.diag(DE))

        invDE[np.isinf(invDE)] = 0  # D_e ^-1
        T = H * (invDE) * H.T
        T += np.mat(np.identity(H.shape[0]))
    else:  # HGNN
        # invDE = np.mat(np.diag(np.power(DE, sigma)))
        invDE = np.mat(np.diag(np.power(DE - np.ones(n_edge), sigma)))
        invDE[np.isinf(invDE)] = 0  # D_e ^-1
        T = H * (invDE) * H.T
        np.fill_diagonal(T, 1)  # 修改其适应数据集
    return T
    # DE = np.mat(np.power(np.diag(nDE)))
    # DE = np.mat(np.power(np.diag(DE)-np.diag(W),sigma))
    # T[range(H.shape[0]),range(H.shape[0])]=0
    DV = np.sum(np.array(T),
                axis=0)  # [2012]数组*是对应位置相乘，矩阵则是矩阵乘法 https://blog.csdn.net/TeFuirnever/article/details/88915383
    invDV = np.power(DV, -0.5)
    invDV[np.isinf(invDV)] = 0
    DV2 = np.mat(np.diag(invDV))  # D_v^-1/2
    G = DV2 * T * DV2
    if 0:
        eign_decomposition(G, plot=True, args=args)
    return G
    #
    # W = np.mat(np.diag(W))
    # H = np.mat(H)
    # HT = H.T
    #
    # if variable_weight:
    #     DV2_H = DV2 * H
    #     invDE_HT_DV2 = invDE * HT * DV2
    #     return DV2_H, W, invDE_HT_DV2
    # else:
    #     G = DV2 * H * W * invDE * HT * DV2  # D_v^1/2 H W D_e^-1 H.T D_v^-1/2
    #     return G


def eign_decomposition(G, plot=False, args=None):
    e_vals, e_vecs = np.linalg.eig(G)
    e_vals = np.sort(e_vals)
    print(e_vals[0], e_vals[-1])
    if plot:
        save = False
        title = 'cocitation-cora'
        xlabel = 'Index'
        ylabel = r'$\lambda$'
        y = e_vals
        x = np.arange(0, len(e_vals))
        plt.plot(x, y, label=fr'$\sigma$ = {args.sigma}', color='b', linewidth=1, linestyle='--')
        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        plt.grid()
        plt.legend()
        if save:
            # plt.savefig(dir_save + title + '.png')
            np.savez(os.path.join('../model/hgnn/npz/' + f"eign_sigma{args.sigma}_self_loop{args.add_self_loop}.npz"),
                     e_vals=e_vals, sigma=args.sigma)

            # plt.savefig(f'./{title}.png', dpi=600, bbox_inches='tight', pad_inches=0.0)
            # plt.savefig(f'./{title}.pdf', dpi=600, bbox_inches='tight', pad_inches=0.0)
        plt.show()
        exit(0)
    return


def symnormalise(M):
    """
    symmetrically normalise sparse matrix
    arguments:
    M: scipy sparse matrix
    returns:
    D^{-1/2} M D^{-1/2}
    where D is the diagonal node-degree matrix
    """

    d = np.array(M.sum(1))

    dhi = np.power(d, -1 / 2).flatten()
    dhi[np.isinf(dhi)] = 0.
    DHI = sp.diags(dhi)  # D half inverse i.e. D^{-1/2}

    return (DHI.dot(M)).dot(DHI)


def ssm2tst(M):
    """
    converts a scipy sparse matrix (ssm) to a torch sparse tensor (tst)
    arguments:
    M: scipy sparse matrix
    returns:
    a torch sparse tensor of M
    """

    M = M.tocoo().astype(np.float32)

    indices = torch.from_numpy(np.vstack((M.row, M.col))).long()
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)

    return torch.sparse.FloatTensor(indices, values, shape)


def normalise(M):
    """
    row-normalise sparse matrix
    arguments:
    M: scipy sparse matrix
    returns:
    D^{-1} M
    where D is the diagonal node-degree matrix
    """

    d = np.array(M.sum(1))

    di = np.power(d, -1).flatten()
    di[np.isinf(di)] = 0.
    DI = sp.diags(di)  # D inverse i.e. D^{-1}

    return DI.dot(M)