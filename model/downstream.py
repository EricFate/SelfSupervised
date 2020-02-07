import torch as t
from torch import nn
from torch.autograd import Variable
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from .base import *
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorboardX import SummaryWriter


class BaseDownStream:
    def __init__(self, extractor, opt):
        self.extractor = extractor
        self.opt = opt
        self.set_experiment_dir(opt["exp_dir"])
        self.writer = SummaryWriter(os.path.join(opt["exp_dir"], 'runs'), 'downstream model %s' % self.__class__)
        self.set_log_file_handler()
        self.init_model()
        self.tensors = {}
        self.allocate_tensors()

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

    def run(self, dataset_train, dataset_test):
        visual = self.opt['visual'] if 'visual' in self.opt else False
        # load dataset from dataloader
        data_loader_train = self.construct_loader(dataset_train)
        data_loader_test = self.construct_loader(dataset_test)

        if visual:
            self.logger.info('==> visualize train set')
            self.visualize_embedding(data_loader_train, 'train set embedding')
            self.logger.info('==> visualize test set')
            self.visualize_embedding(data_loader_test, 'test set embedding')
        # train model if necessary(fine-tune or top)
        self.logger.info('==> train model %s' % self.__class__)
        train_stats = self.train_model(data_loader_train)
        self.logger.info('==> train finished')
        self.logger.info('==> train stats : %s' % train_stats)
        # evaluate model
        self.logger.info('==> evaluate model')
        evaluation_stats = self.evaluate_model(data_loader_test)
        self.logger.info('==> Results: %s' % evaluation_stats)
        # if len(self.optimizers) == 0:
        #     self.init_all_optimizers()
        # self.epoch_size = len(dataset_train)

    def load_to_gpu(self):
        self.extractor = self.extractor.cuda()
        for key, tensor in self.tensors.items():
            self.tensors[key] = tensor.cuda()

    def construct_loader(self, dataset):
        return getattr(dloader, self.opt['loader'], None)(dataset, **self.opt['loader_args'])

    def train_model(self, data_loader_train):
        raise NotImplementedError

    def evaluate_model(self, data_loader_test):
        record = {}
        preds = []
        labels = []
        for data, label in tqdm(data_loader_test()):
            with t.no_grad():
                self.tensors["dataX"].resize_(data.size()).copy_(data)
                feats = self.extractor.extract_feature(self.tensors["dataX"])
                preds.append(self.predict(feats))
                labels.append(label.numpy())
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        record['acc'] = accuracy_score(preds, labels)
        return record

    def predict(self, data):
        raise NotImplementedError

    def init_model(self):
        raise NotImplementedError

    def visualize_embedding(self, data_loader, title=''):
        with t.no_grad():
            feats, labels = self.load_all_features(data_loader)
        self.logger.info('==> load data finished')
        tsne = TSNE(n_components=2, init='pca', random_state=501, verbose=1)
        feats_tsne = tsne.fit_transform(feats)
        self.logger.info('==> tsne finished')
        x_min, x_max = feats_tsne.min(0), feats_tsne.max(0)
        X_norm = (feats_tsne - x_min) / (x_max - x_min)  # normalize
        fig = plt.figure(figsize=(8, 8))
        for i in range(X_norm.shape[0]):
            plt.scatter(X_norm[i, 0], X_norm[i, 1]
                        )
        fig.title(title)
        plt.savefig(fig, fname=os.path.join(self.vis_dir, title + '.png'))

    def load_all_features(self, data_loader):
        feats = []
        labels = []
        for data, label in tqdm(data_loader()):
            self.tensors["dataX"].resize_(data.size()).copy_(data)
            feats.append(self.extractor.extract_feature(self.tensors["dataX"]).cpu().detach().numpy())
            labels.append(label.numpy())
        feats = np.concatenate(feats)
        labels = np.concatenate(labels)
        return feats, labels

    def allocate_tensors(self):
        self.tensors["dataX"] = t.FloatTensor()


class NeuralNetworkDownStream(BaseDownStream):
    def predict(self, data):
        return self.model(data).cpu().detach().numpy()

    def init_model(self):
        self.model = importlib.import_module('networks.%s' % self.opt['net_def']).create_model(**self.opt['model_opt'])

    def train_model(self, data_loader_train):
        for self.curr_epoch in range(self.max_num_epochs):
            self.logger.info(
                "Training epoch [%3d / %3d]" % (self.curr_epoch + 1, self.max_num_epochs))
            train_stats = self.run_train_epoch(
                data_loader_train, self.curr_epoch)
            self.logger.info("==> Training stats: %s" % (train_stats))

    def run_train_epoch(self, data_loader, epoch):
        self.logger.info("Training: %s" % os.path.basename(self.exp_dir))
        self.dloader = data_loader
        self.dataset_train = data_loader.dataset

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


class LogisticDownStream(BaseDownStream):

    def predict(self, data):
        return self.model.predict(data.cpu().detach().numpy())

    def train_model(self, data_loader_train):
        feats, labels = self.load_all_features(data_loader_train)
        X = feats
        Y = labels
        self.model.fit(X, Y)
        record = {}
        record['acc'] = self.model.score(X, Y)
        return record

    def init_model(self):
        self.model = LogisticRegression(**self.opt['model_opt'])
