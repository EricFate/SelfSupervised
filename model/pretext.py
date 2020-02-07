from torch import nn
from data.dataloader import *
from .base import *
import criterions
from utils.color_layers import NNEncLayer, PriorBoostLayer, NonGrayMaskLayer


class RotNet(ClassificationModel):

    def construct_loader(self, dataset):
        return RotateDataLoader(dataset, **self.opt['loader_args'])

    def __init__(self, opt):
        super(RotNet, self).__init__(opt)


class Jigsaw(ClassificationModel):
    def construct_loader(self, dataset):
        return JigsawDataLoader(dataset, **self.opt['loader_args'])


class AMDIM(ClassificationModel):

    def allocate_tensors(self):
        self.tensors['X1'] = t.FloatTensor()
        self.tensors['X2'] = t.FloatTensor()

    def process_batch(self, batch, do_train=True):
        # *************** LOAD BATCH (AND MOVE IT TO GPU) ********
        start = time.time()
        self.tensors["X1"].resize_(batch[0].size()).copy_(batch[0])
        self.tensors["X2"].resize_(batch[1].size()).copy_(batch[1])
        X1 = self.tensors["X1"]
        X2 = self.tensors["X2"]
        batch_load_time = time.time() - start
        # ********************************************************
        if do_train:
            self.optimizers['model'].zero_grad()

        # ********************************************************
        start = time.time()
        # ********************************************************

        # ***************** SET TORCH VARIABLES ******************
        X1_var = t.autograd.Variable(X1, volatile=False)
        X2_var = t.autograd.Variable(X2, volatile=False)
        # ********************************************************

        # ************ FORWARD THROUGH NET ***********************
        losses = self.networks['model'](X1_var, X2_var)
        # ********************************************************

        # *************** COMPUTE LOSSES *************************
        # compute costs for all self-supervised tasks
        loss_g2l = (losses['g2l_1t5'] +
                    losses['g2l_1t7'] +
                    losses['g2l_5t5'])
        loss_inf = loss_g2l + losses['lgt_reg']
        record = {}
        record['g2l_1t5'] = losses['g2l_1t5']
        record['g2l_1t7'] = losses['g2l_1t7']
        record['g2l_5t5'] = losses['g2l_5t5']
        record['lgt_reg'] = losses['lgt_reg']
        record["loss"] = loss_inf.item()
        # ********************************************************

        # ****** BACKPROPAGATE AND APPLY OPTIMIZATION STEP *******
        if do_train:
            loss_inf.backward()
            self.optimizers['model'].step()
        # ********************************************************
        batch_process_time = time.time() - start
        total_time = batch_process_time + batch_load_time
        record["load_time"] = 100 * (batch_load_time / total_time)
        record["process_time"] = 100 * (batch_process_time / total_time)

        return record


class Examplar(ClassificationModel):

    def __init__(self, opt):
        super().__init__(opt)

    def init_all_criterions(self):
        pass

    def before_train(self):
        criterions_defs = self.opt['criterions']
        self.criterions = {}
        for key, val in criterions_defs.items():
            crit_type = val['ctype']
            crit_opt = val['opt'] if ('opt' in val) else None
            self.logger.info('Initialize criterion[%s]: %s with options: %s' % (key, crit_type, crit_opt))
            self.criterions[key] = self.init_criterion(crit_type, crit_opt)

    def init_criterion(self, ctype, copt):
        copt['nLem'] = len(self.dataset_train)
        return getattr(criterions, ctype)(**copt)

    def process_batch(self, batch, do_train=True):
        # *************** LOAD BATCH (AND MOVE IT TO GPU) ********
        start = time.time()
        self.tensors["dataX"].resize_(batch[0].size()).copy_(batch[0])
        self.tensors["labels"].resize_(batch[1].size()).copy_(batch[1])
        dataX = self.tensors["dataX"]
        index = self.tensors["labels"]
        batch_load_time = time.time() - start
        # ********************************************************

        # ********************************************************
        start = time.time()
        if do_train:  # zero the gradients
            self.optimizers['model'].zero_grad()
        # ********************************************************

        # ***************** SET TORCH VARIABLES ******************
        dataX_var = t.autograd.Variable(dataX, volatile=(not do_train))
        index_var = t.autograd.Variable(index, requires_grad=False)
        # ********************************************************

        # ************ FORWARD THROUGH NET ***********************
        feature = self.networks['model'](dataX_var)
        # ********************************************************

        # *************** COMPUTE LOSSES *************************
        record = {}
        output = self.criterions['lemniscate'](feature, index_var)
        loss = self.criterions['nce'](output, index_var)
        record["loss"] = loss.item()
        # ********************************************************

        # ****** BACKPROPAGATE AND APPLY OPTIMIZATION STEP *******
        if do_train:
            loss.backward()
            self.optimizers['model'].step()
        # ********************************************************
        batch_process_time = time.time() - start
        total_time = batch_process_time + batch_load_time
        record["load_time"] = 100 * (batch_load_time / total_time)
        record["process_time"] = 100 * (batch_process_time / total_time)

        return record


class Colorization(ClassificationModel):

    def __init__(self, opt):
        super(Colorization, self).__init__(opt)
        self.encode_layer = NNEncLayer()
        self.boost_layer = PriorBoostLayer()
        self.nongray_mask = NonGrayMaskLayer()

    def allocate_tensors(self):
        self.tensors['image'] = t.FloatTensor()
        self.tensors['image_ab'] = t.FloatTensor()
        self.tensors['targets'] = t.LongTensor()
        self.tensors['boost'] = t.FloatTensor()
        self.tensors['mask'] = t.FloatTensor()

    def process_batch(self, batch, do_train=True):
        start = time.time()
        self.tensors["image"].resize_(batch[0].size()).copy_(batch[0])
        self.tensors["image_ab"].resize_(batch[1].size()).copy_(batch[1])
        image = self.tensors['image']
        image_ab = batch[1]
        batch_load_time = time.time() - start
        # ********************************************************

        # ********************************************************
        start = time.time()
        if do_train:  # zero the gradients
            self.optimizers['model'].zero_grad()
        # ********************************************************

        # ***************** SET TORCH VARIABLES ******************
        image = t.autograd.Variable(image)
        # ********************************************************

        # ************ FORWARD THROUGH NET ***********************
        encode, max_encode = self.encode_layer.forward(image_ab)
        targets = torch.Tensor(max_encode).long()
        # print('set_tar',set(targets[0].cpu().data.numpy().flatten()))
        boost = torch.Tensor(self.boost_layer.forward(encode)).float()
        mask = torch.Tensor(self.nongray_mask.forward(image_ab)).float()
        self.tensors["targets"].resize_(targets.size()).copy_(targets)
        self.tensors["boost"].resize_(boost.size()).copy_(boost)
        self.tensors["mask"].resize_(mask.size()).copy_(mask)

        boost_nongray = self.tensors["boost"] * self.tensors["mask"]
        outputs = self.networks['model'](image)  # .log()
        # ********************************************************

        # *************** COMPUTE LOSSES *************************
        record = {}
        loss = (self.criterions['loss'](outputs, self.tensors["targets"]) * (boost_nongray.squeeze(1))).mean()
        record["loss"] = loss.item()
        # ********************************************************

        # ****** BACKPROPAGATE AND APPLY OPTIMIZATION STEP *******
        if do_train:
            loss.backward()
            self.optimizers['model'].step()
        # ********************************************************
        batch_process_time = time.time() - start
        total_time = batch_process_time + batch_load_time
        record["load_time"] = 100 * (batch_load_time / total_time)
        record["process_time"] = 100 * (batch_process_time / total_time)

        return record
