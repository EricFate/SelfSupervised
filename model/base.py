import importlib
from tqdm import tqdm
import os
import time

import logging
import datetime
import torch as t
from torch import nn
from data.utils import accuracy
from .meter import DAverageMeter

import data.dataloader as dloader
from tensorboardX import SummaryWriter


class BaseModel(nn.Module):
    def forward(self, x):
        return self.networks['model'](x)

    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.set_experiment_dir(opt["exp_dir"])
        self.writer = SummaryWriter(os.path.join(opt["exp_dir"], 'runs'), 'pretext model %s' % self.__class__)
        self.set_log_file_handler()

        self.tensors = {}
        self.optimizers = {}
        self.networks = {}
        self.optimizers = {}
        self.criterions = {}
        self.logger.info("Algorithm options %s" % opt)
        self.opt = opt
        self.init_all_networks()
        self.init_all_criterions()
        self.allocate_tensors()
        self.curr_epoch = 0
        # self.init_optimizer(opt['optimizer'])

        self.keep_best_model_metric_name = opt["best_metric"] if ("best_metric" in opt) else None

    def init_all_criterions(self):
        criterions_defs = self.opt['criterions']
        self.criterions = {}
        for key, val in criterions_defs.items():
            crit_type = val['ctype']
            crit_opt = val['opt'] if ('opt' in val) else None
            self.logger.info('Initialize criterion[%s]: %s with options: %s' % (key, crit_type, crit_opt))
            self.criterions[key] = self.init_criterion(crit_type, crit_opt)

    def init_criterion(self, ctype, copt):
        return getattr(nn, ctype)(copt)

    def set_experiment_dir(self, directory_path):
        self.exp_dir = directory_path
        if not os.path.isdir(self.exp_dir):
            os.makedirs(self.exp_dir)

        self.vis_dir = os.path.join(directory_path, "visuals")
        if not os.path.isdir(self.vis_dir):
            os.makedirs(self.vis_dir)

        self.preds_dir = os.path.join(directory_path, "preds")
        if not os.path.isdir(self.preds_dir):
            os.makedirs(self.preds_dir)

    def set_log_file_handler(self):
        self.logger = logging.getLogger(__name__)

        strHandler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)-8s - %(levelname)-6s - %(message)s")
        strHandler.setFormatter(formatter)
        self.logger.addHandler(strHandler)
        self.logger.setLevel(logging.INFO)

        log_dir = os.path.join(self.exp_dir, "logs")
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        now_str = datetime.datetime.now().__str__().replace(" ", "_")

        self.log_file = os.path.join(log_dir, "LOG_INFO_" + now_str + ".txt")
        self.log_fileHandler = logging.FileHandler(self.log_file)
        self.log_fileHandler.setFormatter(formatter)
        self.logger.addHandler(self.log_fileHandler)

    def init_all_networks(self):
        networks_defs = self.opt["networks"]

        for key, val in networks_defs.items():
            self.logger.info("Set network %s" % key)
            def_file = val["def"]
            net_opt = val["opt"]
            # self.optim_params[key] = val["optim_params"] if ("optim_params" in val) else None
            pretrained_path = val["pretrained"] if ("pretrained" in val) else None
            self.init_network(def_file, net_opt, pretrained_path, key)
            self.init_optimizer(key, val['optim_type'], val['optim_params'])

    def init_record_of_best_model(self):
        self.max_metric_val = None
        self.best_stats = None
        self.best_epoch = None

    def adjust_learning_rates(self, epoch):
        # filter out the networks that are not trainable and that do
        # not have a learning rate Look Up Table (LUT_lr) in their optim_params
        optim_params_filtered = {k: v for k, v in self.optim_params.items()
                                 if (v != None and ('LUT_lr' in v))}

        for key, oparams in optim_params_filtered.items():
            LUT = oparams['LUT_lr']
            lr = next((lr for (max_epoch, lr) in LUT if max_epoch > epoch), LUT[-1][1])
            self.logger.info('==> Set to %s optimizer lr = %.10f' % (key, lr))
            for param_group in self.optimizers[key].param_groups:
                param_group['lr'] = lr

    def init_network(self, net_def_file, net_opt, pretrained_path, key):
        self.logger.info('==> Initiliaze network %s from file %s with opts: %s' % (key, net_def_file, net_opt))
        # if (not os.path.isfile(net_def_file)):
        #     raise ValueError('Non existing file: {0}'.format(net_def_file))

        network = importlib.import_module('networks.%s' % net_def_file).create_model(**net_opt)
        if pretrained_path != None:
            self.load_pretrained(network, pretrained_path)
        self.networks[key] = network

    def load_pretrained(self, network, pretrained_path):
        self.logger.info("==> Load pretrained parameters from file %s:" % (pretrained_path))

        assert os.path.isfile(pretrained_path)
        pretrained_model = t.load(pretrained_path)
        if pretrained_model["network"].keys() == network.state_dict().keys():
            network.load_state_dict(pretrained_model["network"])
        else:
            self.logger.info(
                "==> WARNING: network parameters in pre-trained file %s do not strictly match" % (pretrained_path)
            )
            for pname, param in network.named_parameters():
                if pname in pretrained_model["network"]:
                    self.logger.info("==> Copying parameter %s from file %s" % (pname, pretrained_path))
                    param.data.copy_(pretrained_model["network"][pname])

    def init_optimizer(self, key, optim_type, optim_opts):
        # learning_rate = optim_opts["lr"]
        parameters = filter(lambda p: p.requires_grad, self.networks[key].parameters())
        self.logger.info("Initialize optimizer: %s with params: %s for netwotk: %s" % (optim_type, optim_opts, key))
        if optim_type == "adam":
            optimizer = t.optim.Adam(parameters, **optim_opts)
        elif optim_type == "sgd":
            optimizer = t.optim.SGD(
                parameters,
                **optim_opts
            )
        else:
            raise ValueError("Not supported or recognized optim_type", optim_type)
        self.optimizers[key] = optimizer
        # raise NotImplementedError

    def load_to_gpu(self):
        for key, net in self.networks.items():
            self.networks[key] = net.cuda()

        for key, criterion in self.criterions.items():
            self.criterions[key] = criterion.cuda()

        for key, tensor in self.tensors.items():
            self.tensors[key] = tensor.cuda()

    def save_checkpoint(self, epoch, suffix=''):
        for key, net in self.networks.items():
            if self.optimizers[key] is None:
                continue
            self.save_network(key, epoch, suffix=suffix)
            self.save_optimizer(key, epoch, suffix=suffix)

    def load_checkpoint(self, epoch, train=True, suffix=""):
        self.logger.info("Load checkpoint of epoch %d" % (epoch))

        for key, net in self.networks.items():  # Load networks
            if self.optim_params[key] is None:
                continue
            self.load_network(key, epoch, suffix)

        if train:  # initialize and load optimizers
            self.init_all_optimizers()
            for key, net in self.networks.items():
                if self.optim_params[key] is None:
                    continue
                self.load_optimizer(key, epoch, suffix)

        self.curr_epoch = epoch

    def delete_checkpoint(self, epoch, suffix=""):
        for key, net in self.networks.items():
            if self.optimizers[key] is None:
                continue

            filename_net = self._get_net_checkpoint_filename(key, epoch) + suffix
            if os.path.isfile(filename_net):
                os.remove(filename_net)

            filename_optim = self._get_optim_checkpoint_filename(key, epoch) + suffix
            if os.path.isfile(filename_optim):
                os.remove(filename_optim)

    def save_network(self, net_key, epoch, suffix=""):
        assert net_key in self.networks
        filename = self._get_net_checkpoint_filename(net_key, epoch) + suffix
        state = {"epoch": epoch, "network": self.networks[net_key].state_dict()}
        t.save(state, filename)

    def save_optimizer(self, net_key, epoch, suffix=""):
        assert net_key in self.optimizers
        filename = self._get_optim_checkpoint_filename(net_key, epoch) + suffix
        state = {"epoch": epoch, "optimizer": self.optimizers[net_key].state_dict()}
        t.save(state, filename)

    def load_network(self, net_key, epoch, suffix=""):
        assert net_key in self.networks
        filename = self._get_net_checkpoint_filename(net_key, epoch) + suffix
        assert os.path.isfile(filename)
        if os.path.isfile(filename):
            checkpoint = t.load(filename)
            self.networks[net_key].load_state_dict(checkpoint["network"])

    def load_optimizer(self, net_key, epoch, suffix=""):
        assert net_key in self.optimizers
        filename = self._get_optim_checkpoint_filename(net_key, epoch) + suffix
        assert os.path.isfile(filename)
        if os.path.isfile(filename):
            checkpoint = t.load(filename)
            self.optimizers[net_key].load_state_dict(checkpoint["optimizer"])

    def _get_net_checkpoint_filename(self, net_key, epoch):
        return os.path.join(self.exp_dir, net_key + "_net_epoch" + str(epoch))

    def _get_optim_checkpoint_filename(self, net_key, epoch):
        return os.path.join(self.exp_dir, net_key + "_optim_epoch" + str(epoch))

    def train_step(self, batch):
        raise NotImplementedError

    def init_tensorboard(self, dataloader):
        first = next(iter(dataloader()))
        for k, v in self.networks.items():
            self.writer.add_graph(v, t.rand_like(first[0].cuda()))

    def before_train(self):
        pass

    def solve(self, dataset_train, dataset_test=None, cuda=True):
        self.max_num_epochs = self.opt["max_num_epochs"]
        start_epoch = self.curr_epoch
        # if len(self.optimizers) == 0:
        #     self.init_all_optimizers()
        # self.epoch_size = len(dataset_train)
        self.dataset_train = dataset_train
        self.dataset_eval = dataset_test
        data_loader_train = self.construct_loader(dataset_train)
        data_loader_test = self.construct_loader(dataset_test)
        # self.init_tensorboard(data_loader_train)
        self.before_train()
        if cuda:
            self.load_to_gpu()
        self.init_record_of_best_model()
        for self.curr_epoch in range(start_epoch, self.max_num_epochs):
            self.logger.info(
                "Training epoch [%3d / %3d]" % (self.curr_epoch + 1, self.max_num_epochs))
            train_stats = self.run_train_epoch(
                data_loader_train, self.curr_epoch)
            self.logger.info("==> Training stats: %s" % train_stats)
            self.writer.add_scalars('train_stats', train_stats, self.curr_epoch)
            # self.adjust_learning_rates(self.curr_epoch)

            # create a checkpoint in the current epoch
            self.save_checkpoint(self.curr_epoch + 1)
            if start_epoch != self.curr_epoch:  # delete the checkpoint of the previous epoch
                self.delete_checkpoint(self.curr_epoch)

            if data_loader_test is not None:
                eval_stats = self.evaluate(data_loader_test)
                self.logger.info("==> Evaluation stats: %s" % eval_stats)
                self.keep_record_of_best_model(eval_stats, self.curr_epoch)
                self.writer.add_scalars('test_stats', eval_stats, self.curr_epoch)

        self.print_eval_stats_of_best_model()

    def keep_record_of_best_model(self, eval_stats, current_epoch):
        if self.keep_best_model_metric_name is not None:
            metric_name = self.keep_best_model_metric_name
            if (metric_name not in eval_stats):
                raise ValueError(
                    'The provided metric {0} for keeping the best model is not computed by the evaluation routine.'.format(
                        metric_name))
            metric_val = eval_stats[metric_name]
            if self.max_metric_val is None or metric_val > self.max_metric_val:
                self.max_metric_val = metric_val
                self.best_stats = eval_stats
                self.save_checkpoint(self.curr_epoch + 1, suffix='.best')
                if self.best_epoch is not None:
                    self.delete_checkpoint(self.best_epoch + 1, suffix='.best')
                self.best_epoch = current_epoch
                self.print_eval_stats_of_best_model()

    def evaluate(self, dloader):
        self.logger.info('Evaluating: %s' % os.path.basename(self.exp_dir))
        self.dloader = dloader
        self.logger.info('==> Dataset: %s [%d images]' % (dloader.dataset.name, len(dloader)))
        for key, network in self.networks.items():
            network.eval()

        eval_stats = DAverageMeter()
        self.bnumber = len(dloader())
        for idx, batch in enumerate(tqdm(dloader())):
            self.biter = idx
            eval_stats_this = self.evaluation_step(batch)
            eval_stats.update(eval_stats_this)

        self.logger.info('==> Results: %s' % eval_stats.average())

        return eval_stats.average()

    def print_eval_stats_of_best_model(self):
        if self.best_stats is not None:
            metric_name = self.keep_best_model_metric_name
            self.logger.info('==> Best results w.r.t. %s metric: epoch: %d - %s' % (
                metric_name, self.best_epoch + 1, self.best_stats))

    def run_train_epoch(self, data_loader, epoch):
        self.logger.info("Training: %s" % os.path.basename(self.exp_dir))
        self.dloader = data_loader

        # for key, network in self.networks.items():
        #     if self.optimizers[key] is None:
        #         network.eval()
        #     else:
        #         network.train()

        disp_step = self.opt["disp_step"] if ("disp_step" in self.opt) else 50
        train_stats = DAverageMeter()
        # self.bnumber = len(data_loader)
        for idx, batch in enumerate(tqdm(data_loader(epoch))):
            self.biter = idx
            train_stats_this = self.train_step(batch)
            train_stats.update(train_stats_this)
            if (idx + 1) % disp_step == 0:
                self.logger.info(
                    "==> Iteration [%d][%d / %d]: %s"
                    % (epoch + 1, idx + 1, len(data_loader), str(train_stats.average()))
                )
        return train_stats.average()

    def construct_loader(self, dataset):
        return getattr(dloader, self.opt['loader'], None)(dataset, **self.opt['loader_args'])

    def allocate_tensors(self):
        """(Optional) allocate torch tensors that could potentially be used in
            in the train_step() or evaluation_step() functions. If the
            load_to_gpu() function is called then those tensors will be moved to
            the gpu device.
        """
        raise NotImplementedError


class ClassificationModel(BaseModel):
    # def init_network(self):
    #     self.networks['model'] = resnet18()
    #
    # def init_optimizer(self, optim_opts):
    #     self.optimizers['model'] = t.optim.Adam(self.networks['model'].parameters(), **optim_opts)

    def allocate_tensors(self):
        self.tensors["dataX"] = t.FloatTensor()
        self.tensors["labels"] = t.LongTensor()

    def train_step(self, batch):
        return self.process_batch(batch, do_train=True)

    def evaluation_step(self, batch):
        return self.process_batch(batch, do_train=False)

    def process_batch(self, batch, do_train=True):
        # *************** LOAD BATCH (AND MOVE IT TO GPU) ********
        start = time.time()
        self.tensors["dataX"].resize_(batch[0].size()).copy_(batch[0])
        self.tensors["labels"].resize_(batch[1].size()).copy_(batch[1])
        dataX = self.tensors["dataX"]
        labels = self.tensors["labels"]
        batch_load_time = time.time() - start
        # ********************************************************

        # ********************************************************
        start = time.time()
        if do_train:  # zero the gradients
            self.optimizers['model'].zero_grad()
        # ********************************************************

        # ***************** SET TORCH VARIABLES ******************
        dataX_var = t.autograd.Variable(dataX, volatile=(not do_train))
        labels_var = t.autograd.Variable(labels, requires_grad=False)
        # ********************************************************

        # ************ FORWARD THROUGH NET ***********************
        pred_var = self.networks['model'](dataX_var)
        # ********************************************************

        # *************** COMPUTE LOSSES *************************
        record = {}
        loss_total = self.criterions['loss'](pred_var, labels_var)
        record["prec1"] = accuracy(pred_var.data, labels, topk=(1,))[0].item()
        record["loss"] = loss_total.item()
        # ********************************************************

        # ****** BACKPROPAGATE AND APPLY OPTIMIZATION STEP *******
        if do_train:
            loss_total.backward()
            self.optimizers['model'].step()
        # ********************************************************
        batch_process_time = time.time() - start
        total_time = batch_process_time + batch_load_time
        record["load_time"] = 100 * (batch_load_time / total_time)
        record["process_time"] = 100 * (batch_process_time / total_time)

        return record
