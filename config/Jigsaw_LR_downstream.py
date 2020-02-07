config = {}

config['model'] = 'LogisticDownStream'
model_opt = {}
opt = {'model_opt': model_opt, 'loader': 'OriginDataLoader'}
opt['loader_args'] = {}
config['opt'] = opt

config['pretext'] = {'config': 'JigsawNet_ResNet', 'net_key': 'model'}
