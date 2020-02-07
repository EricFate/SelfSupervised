import torch as t
import argparse
from data import GenericDataset
import model.downstream as downstream
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
parser.add_argument('--config', type=str, default='RotNet_LR_downstream',
                    help='config file')

args = parser.parse_args()

if __name__ == "__main__":
    cuda = False
    if args.cuda and t.cuda.is_available():
        print('gpu %s available'%args.gpu)
        cuda = True
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # if args_opt.semi == -1:
    opt = importlib.import_module('config.%s' % args.config).config
    # *************load extractor***************************
    pretext = opt['pretext']
    exp_directory = os.path.join('.', 'experiments', pretext['config'])

    pretext_opt = importlib.import_module('config.%s' % pretext['config']).config
    model_opt = pretext_opt['networks'][pretext['net_key']]
    network = importlib.import_module('networks.%s' % model_opt['def']).create_model(**model_opt['opt'])
    model_ckpt = os.path.join(exp_directory, pretext['net_key'] + "_net_epoch" + str(pretext_opt['max_num_epochs']))
    network.load_state_dict(t.load(model_ckpt)['network'])

    # opt['opt']['loader'].update(pretext_opt['loader'])
    opt['opt']['loader_args'].update(pretext_opt['loader_args'])
    opt['opt']['exp_dir'] = os.path.join('.', 'experiments', args.config)
    downstream_model = getattr(downstream, opt['model'])(network, opt['opt'])
    if cuda:
        downstream_model.load_to_gpu()
    dataset_train = GenericDataset(
        dataset_name=args.dataset,
        split='train')
    dataset_test = GenericDataset(
        dataset_name=args.dataset,
        split='val')
    print('run downstream task for %s',opt['pretext'])
    downstream_model.run(dataset_train, dataset_test)
