batch_size = 128

config = {}
# set the parameters related to the training and testing set
# data_train_opt = {}
# data_train_opt['batch_size'] = batch_size
# data_train_opt['unsupervised'] = True
# data_train_opt['epoch_size'] = None
# data_train_opt['random_sized_crop'] = False
# data_train_opt['dataset_name'] = 'cifar10'
# data_train_opt['split'] = 'train'
#
# data_test_opt = {}
# data_test_opt['batch_size'] = batch_size
# data_test_opt['unsupervised'] = True
# data_test_opt['epoch_size'] = None
# data_test_opt['random_sized_crop'] = False
# data_test_opt['dataset_name'] = 'cifar10'
# data_test_opt['split'] = 'test'
#
# config['data_train_opt'] = data_train_opt
# config['data_test_opt'] = data_test_opt
# config['max_num_epochs'] = 200
#
# net_opt = {}
# net_opt['num_classes'] = 4
# net_opt['num_stages']  = 4
# net_opt['use_avg_on_conv3'] = False
#
# networks = {}
# net_optim_params = {'optim_type': 'sgd', 'lr': 0.1, 'momentum':0.9, 'weight_decay': 5e-4, 'nesterov': True, 'LUT_lr':[(60, 0.1),(120, 0.02),(160, 0.004),(200, 0.0008)]}
# networks['model'] = {'def_file': 'architectures/NetworkInNetwork.py', 'pretrained': None, 'opt': net_opt,  'optim_params': net_optim_params}
# config['networks'] = networks
#
#
#
low_dim = 128
criterions = {}
lem_opt = {'inputSize': low_dim, 'nLem': 0, 'K': 4096, 'T': 0.07, 'momentum': 0.5}
criterions['lemniscate'] = {'ctype': 'NCEAverage', 'opt': lem_opt}
nce_opt = {'nLem': 0}
criterions['nce'] = {'ctype': 'NCECriterion', 'opt': nce_opt}
config['criterions'] = criterions
# config['algorithm_type'] = 'ClassificationModel'
net_opt = {}
net_opt['low_dim'] = low_dim
networks = {}
optim = {'lr': 1e-4, 'weight_decay': 5e-4}
networks['model'] = {'def': 'ResNetFromExamplar', 'optim_type': 'adam', 'pretrained': None, 'opt': net_opt,
                     'optim_params': optim}
config['networks'] = networks

config['model'] = 'Examplar'
config['max_num_epochs'] = 50
config['loader_args'] = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}
config['loader'] = 'ExamplarDataLoader'
config['dataset'] = {}
# *************downstream config****************
