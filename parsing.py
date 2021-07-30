import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        help='Dataset')
    parser.add_argument('--epoch', type=int, default=40,
                        help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=64,
                        help='Node dimension')
    parser.add_argument('--num_channels', type=int, default=2,
                        help='number of channels')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='l2 reg')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layer')
    parser.add_argument('--norm', type=str, default='true',
                        help='normalization')
    parser.add_argument('--adaptive_lr', type=str, default='false',
                        help='adaptive learning rate')
    # argments for mpnn-r
    parser.add_argument('--data', type=str, default="dblp", help='data name (FB-AUTO)')
    parser.add_argument('--split', type=str, default=1, help='train-test split used for the dataset')

    # argment for HGNN
    parser.add_argument('--add_self_loop', action='store_true', default=False, help='Don"t add self loop in adj')  # gcnii*
    parser.add_argument('--save_file', default='results.csv', help='save file name')
    parser.add_argument('--type', default='mpnn', help='model type')

    parser.add_argument('--degree', type=int, default=16,
                        help='degree of the approximation.')
    parser.add_argument('--sgc_alpha', type=float, default=0.05,
                        help='alpha.')
    parser.add_argument('--sgc_belta', type=float, default=0.05,
                        help='belta.')
    parser.add_argument('--sigma', type=float, default=-1,
                        help='sigma for transition matirx.')
    parser.add_argument('--learnable', action='store_true', default=False, help='learnable  coefficient')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    args = parser.parse_args()
    return args