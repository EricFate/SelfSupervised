batch_size = 128

config = {}

criterions = {}
criterions['loss'] = {'ctype': 'CrossEntropyLoss', 'opt': None}
config['criterions'] = criterions
# config['algorithm_type'] = 'ClassificationModel'
net_opt = {}
net_opt['ndf'] = 128
net_opt['n_rkhs'] = 1024
net_opt['n_depth'] = 3
net_opt['num_channels'] = 1
networks = {}
optim = {'lr': 2e-4, 'weight_decay': 1e-5, 'betas': (0.8, 0.999)}
networks['model'] = {'def': 'ColorNet', 'optim_type': 'adam', 'pretrained': None, 'opt': net_opt,
                     'optim_params': optim}
config['networks'] = networks

config['model'] = 'Colorization'
config['max_num_epochs'] = 50
config['loader_args'] = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}
config['loader'] = 'ColorDataLoader'
config['dataset'] = {}
