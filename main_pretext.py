import torch as t
import argparse
from data import GenericDataset
import model.pretext as pretext
import config
import os
import importlib

parser = argparse.ArgumentParser()
parser.add_argument('--evaluate', default=False, action='store_true')
parser.add_argument('--checkpoint', type=int, default=0,
                    help='checkpoint (epoch id) that will be loaded')
parser.add_argument('--num_workers', type=int, default=0,
                    help='number of data loading workers')
parser.add_argument('--cuda', type=bool,
                    default=True, help='enables cuda')
parser.add_argument('--gpu', type=str,
                    default='1', help='visible gpu')
parser.add_argument('--disp_step', type=int, default=50,
                    help='display step during training')
parser.add_argument('--dataset', type=str, default='tinyimagenet',
                    help='dataset for training')
parser.add_argument('--config', type=str, default='RotNet_Resnet18',
                    help='config file')

args = parser.parse_args()

if __name__ == "__main__":
    cuda = False
    if args.cuda and t.cuda.is_available():
        print('gpu %s available' % args.gpu)
        cuda = True
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # if args_opt.semi == -1:
    exp_directory = os.path.join('.', 'experiments', args.config)
    opt = importlib.import_module('config.%s' % args.config).config
    opt['exp_dir'] = exp_directory
    dataset_train = GenericDataset(
        dataset_name=args.dataset,
        split='train',
        **opt['dataset'])
    dataset_test = GenericDataset(
        dataset_name=args.dataset,
        split='val',
        **opt['dataset'])
    model = getattr(pretext, opt['model'], None)(opt)
    if model is not None:
        model.solve(dataset_train, dataset_test, cuda)
