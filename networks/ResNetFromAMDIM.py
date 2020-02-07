import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.AMDIMutils import flatten, Flatten
from .costs import LossMultiNCE
from .EncodingBlocks import NopNet, FakeRKHSConvNet, ConvResNxN, ConvResBlock, ResEncoder


class Encoder(nn.Module):
    def __init__(self, dummy_batch, num_channels=3, ndf=64, n_rkhs=512,
                 n_depth=3, encoder_size=32, use_bn=False):
        super(Encoder, self).__init__()
        self.ndf = ndf
        self.n_rkhs = n_rkhs
        self.use_bn = use_bn
        self.dim2layer = None

        self.layer_list = ResEncoder(num_channels, ndf, n_rkhs,
                                     n_depth, encoder_size, use_bn).layer_list
        # encoding block for local features
        print('Using a {}x{} encoder'.format(encoder_size, encoder_size))

        self._config_modules(dummy_batch, [1, 5, 7], n_rkhs, use_bn)

    def init_weights(self, init_scale=1.):
        '''
        Run custom weight init for modules...
        '''
        for layer in self.layer_list:
            if isinstance(layer, (ConvResNxN, ConvResBlock)):
                layer.init_weights(init_scale)
        for layer in self.modules():
            if isinstance(layer, (ConvResNxN, ConvResBlock)):
                layer.init_weights(init_scale)
            if isinstance(layer, FakeRKHSConvNet):
                layer.init_weights(init_scale)

    def _config_modules(self, x, rkhs_layers, n_rkhs, use_bn):
        '''
        Configure the modules for extracting fake rkhs embeddings for infomax.
        '''
        enc_acts = self._forward_acts(x)
        self.dim2layer = {}
        for i, h_i in enumerate(enc_acts):
            for d in rkhs_layers:
                if h_i.size(2) == d:
                    self.dim2layer[d] = i
        # get activations and feature sizes at different layers
        self.ndf_1 = enc_acts[self.dim2layer[1]].size(1)
        self.ndf_5 = enc_acts[self.dim2layer[5]].size(1)
        self.ndf_7 = enc_acts[self.dim2layer[7]].size(1)
        # configure modules for fake rkhs embeddings
        self.rkhs_block_1 = NopNet()
        self.rkhs_block_5 = FakeRKHSConvNet(self.ndf_5, n_rkhs, use_bn)
        self.rkhs_block_7 = FakeRKHSConvNet(self.ndf_7, n_rkhs, use_bn)

    def _forward_acts(self, x):
        '''
        Return activations from all layers.
        '''
        # run forward pass through all layers
        layer_acts = [x]
        for _, layer in enumerate(self.layer_list):
            layer_in = layer_acts[-1]
            layer_out = layer(layer_in)
            layer_acts.append(layer_out)
        # remove input from the returned list of activations
        return_acts = layer_acts[1:]
        return return_acts

    def forward(self, x):
        '''
        Compute activations and Fake RKHS embeddings for the batch.
        '''
        # compute activations in all layers for x
        acts = self._forward_acts(x)
        # gather rkhs embeddings from certain layers
        r1 = self.rkhs_block_1(acts[self.dim2layer[1]])
        r5 = self.rkhs_block_5(acts[self.dim2layer[5]])
        r7 = self.rkhs_block_7(acts[self.dim2layer[7]])
        return r1, r5, r7


class Model(nn.Module):
    def __init__(self, ndf, n_rkhs, tclip=20.,
                 n_depth=3, encoder_size=32, use_bn=False):
        super(Model, self).__init__()
        self.hyperparams = {
            'ndf': ndf,
            'n_rkhs': n_rkhs,
            'tclip': tclip,
            'n_depth': n_depth,
            'encoder_size': encoder_size,
            'use_bn': use_bn
        }

        # self.n_rkhs = n_rkhs
        self.tasks = ('1t5', '1t7', '5t5', '5t7', '7t7')
        dummy_batch = torch.zeros((2, 3, encoder_size, encoder_size))

        # encoder that provides multiscale features
        self.encoder = Encoder(dummy_batch, num_channels=3, ndf=ndf,
                               n_rkhs=n_rkhs, n_depth=n_depth,
                               encoder_size=encoder_size, use_bn=use_bn)
        rkhs_1, rkhs_5, _ = self.encoder(dummy_batch)
        # convert for multi-gpu use
        self.encoder = nn.DataParallel(self.encoder)

        # configure hacky multi-gpu module for infomax costs
        self.g2l_loss = LossMultiNCE(tclip=tclip)

        # configure modules for classification with self-supervised features

        # gather lists of self-supervised and classifier modules
        self.info_modules = [self.encoder.module, self.g2l_loss]

    def init_weights(self, init_scale=1.):
        self.encoder.module.init_weights(init_scale)

    def encode(self, x, no_grad=True, use_eval=False):
        '''
        Encode the images in x, with or without grads detached.
        '''
        if use_eval:
            self.eval()
        if no_grad:
            with torch.no_grad():
                rkhs_1, rkhs_5, rkhs_7 = self.encoder(x)
        else:
            rkhs_1, rkhs_5, rkhs_7 = self.encoder(x)
        if use_eval:
            self.train()
        return rkhs_1, rkhs_5, rkhs_7

    # def reset_evaluator(self, n_classes=None):
    #     '''
    #     Reset the evaluator module, e.g. to apply encoder on new data.
    #     - evaluator is reset to have n_classes classes (if given)
    #     '''
    #     dim_1 = self.evaluator.dim_1
    #     if n_classes is None:
    #         n_classes = self.evaluator.n_classes
    #     self.class_modules = [self.evaluator]
    #     return self.evaluator

    def forward(self, x1, x2):
        '''
        Input:
          x1 : images from which to extract features -- x1 ~ A(x)
          x2 : images from which to extract features -- x2 ~ A(x)
          class_only : whether we want all outputs for infomax training
        Output:
          res_dict : various outputs depending on the task
        '''
        # dict for returning various values
        res_dict = {}
        # run augmented image pairs through the encoder
        r1_x1, r5_x1, r7_x1 = self.encoder(x1)
        r1_x2, r5_x2, r7_x2 = self.encoder(x2)

        # compute NCE infomax objective at multiple scales
        loss_1t5, loss_1t7, loss_5t5, lgt_reg = \
            self.g2l_loss(r1_x1, r5_x1, r7_x1, r1_x2, r5_x2, r7_x2)
        res_dict['g2l_1t5'] = loss_1t5
        res_dict['g2l_1t7'] = loss_1t7
        res_dict['g2l_5t5'] = loss_5t5
        res_dict['lgt_reg'] = lgt_reg
        # grab global features for use elsewhere
        res_dict['rkhs_glb'] = flatten(r1_x1)

        return res_dict

    def extract_feature(self, X):
        # run augmented image pairs through the encoder
        r1_x1, r5_x1, r7_x1 = self.encoder(X)
        return flatten(r1_x1)


def create_model(**kwargs):
    return Model(**kwargs)
