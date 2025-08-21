from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch


class BDGP(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'BDGP.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'BDGP.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path+'BDGP.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()


class CCV(Dataset):
    def __init__(self, path):
        self.data1 = np.load(path+'STIP.npy').astype(np.float32)
        scaler = MinMaxScaler()
        self.data1 = scaler.fit_transform(self.data1)
        self.data2 = np.load(path+'SIFT.npy').astype(np.float32)
        self.data3 = np.load(path+'MFCC.npy').astype(np.float32)
        self.labels = np.load(path+'label.npy')

    def __len__(self):
        return 6773

    def __getitem__(self, idx):
        x1 = self.data1[idx]
        x2 = self.data2[idx]
        x3 = self.data3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(
           x2), torch.from_numpy(x3)], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class MNIST_USPS(Dataset):
    def __init__(self, path, V, n):
        self.view = V
        self.n = n
        self.dataset = scipy.io.loadmat(path + 'MNIST_USPS.mat')
        self.Y = self.dataset['Y'].astype(np.int32).reshape(n,)
        self.X1 = self.dataset['X1'].astype(np.float32).reshape(n, -1, 28, 28)
        self.X2 = self.dataset['X2'].astype(np.float32).reshape(n, -1, 28, 28)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return [self.X1[idx], self.X2[idx]], self.Y[idx], idx


class Fashion(Dataset):
    def __init__(self, path, V, n):
        self.view = V
        self.n = n
        self.dataset = scipy.io.loadmat(path + 'Fashion.mat')
        self.Y = self.dataset['Y'].astype(np.int32).reshape(n,)
        self.X1 = self.dataset['X1'].astype(np.float32).reshape(n, -1, 28, 28)
        self.X2 = self.dataset['X2'].astype(np.float32).reshape(n, -1, 28, 28)
        self.X3 = self.dataset['X3'].astype(np.float32).reshape(n, -1, 28, 28)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return [self.X1[idx], self.X2[idx], self.X3[idx]], self.Y[idx], idx


class Caltech(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(
                self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
                self.view5[idx]), torch.from_numpy(self.view4[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx]), torch.from_numpy(
                self.view4[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


def load_data(dataset):
    if dataset == "BDGP":
        dataset = BDGP('../../Datasets/Multi_View/')
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5
    elif dataset == "MNIST_USPS":
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 5000
        dataset = MNIST_USPS('../../Datasets/Multi_View/', view, data_size)
    elif dataset == "CCV":
        dataset = CCV('../../Datasets/Multi_View/CCV/')
        dims = [5000, 5000, 4000]
        view = 3
        data_size = 6773
        class_num = 20
    elif dataset == "Fashion":
        dims = [784, 784, 784]
        view = 3
        data_size = 10000
        class_num = 10
        dataset = Fashion('../../Datasets/Multi_View/', view, data_size)
    elif dataset == "Caltech_2V":
        dataset = Caltech('../../Datasets/Multi_View/Caltech-5V.mat', view=2)
        dims = [40, 254]
        view = 2
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech_3V":
        dataset = Caltech('../../Datasets/Multi_View/Caltech-5V.mat', view=3)
        dims = [40, 254, 928]
        view = 3
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech_4V":
        dataset = Caltech('../../Datasets/Multi_View/Caltech-5V.mat', view=4)
        dims = [40, 254, 928, 512]
        view = 4
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech_5V":
        dataset = Caltech('../../Datasets/Multi_View/Caltech-5V.mat', view=5)
        dims = [40, 254, 928, 512, 1984]
        view = 5
        data_size = 1400
        class_num = 7
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
