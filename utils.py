import torch
import os
import random
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", output_file=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.output_file = output_file

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def tsne_plot(embeddings, labels, path='./'):
    views = len(embeddings)
    n_clusters = len(np.unique(labels))
    for v in range(views):
        file = os.path.join(path, 'tsne_V%d.svg' %v)
        tsne = TSNE(n_components=2, random_state=0)
        X_2d = tsne.fit_transform(embeddings[v])

        plt.figure(figsize=(6, 5))
        colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
        for i in range(n_clusters):
            plt.scatter(X_2d[labels==i, 0], X_2d[labels==i, 1], s=5, color=colors[i], label=str(i))
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.savefig(file)


def metrics_plot(embeddings, path='./'):
    views = len(embeddings)
    for v in range(views):
        file = os.path.join(path, 'metrics_V%d.svg' % v)
        sim = np.dot(embeddings[v].T, embeddings[v])
        plt.figure(figsize=(8, 6))
        plt.imshow(sim, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig(file)