config = {}

config['model'] = 'LogisticDownStream'
model_opt = {'solver': 'lbfgs', 'max_iter': 500, 'verbose': 1}
opt = {'model_opt': model_opt, 'loader': 'OriginDataLoader', 'visual': False}
opt['loader_args'] = {}
config['opt'] = opt

config['pretext'] = {'config': 'RotNet_ResBlocksNet_cifar10', 'net_key': 'model'}
