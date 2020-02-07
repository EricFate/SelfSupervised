config = {}

config['model'] = 'LogisticDownStream'
model_opt = {'solver': 'lbfgs', 'max_iter': 500, 'verbose': 1}
opt = {'model_opt': model_opt, 'loader': 'ColorDataLoader', 'visual': False}
opt['loader_args'] = {'train':False}
config['opt'] = opt

config['pretext'] = {'config': 'Color_ResBlocks_cifar10', 'net_key': 'model'}
