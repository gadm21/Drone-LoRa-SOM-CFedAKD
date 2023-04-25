

import sys 
import os 
import pandas as pd
import numpy as np 
sys.path.append('..')

# from data.depth_dataset import data_pre as depth_pre
# from data.large_scale_HARBox import data_pre as harbox_pre
# from data.imu_dataset import data_pre as imu_pre

import torch
import torch.utils.data as torch_data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST
import torchvision
from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sklearn.model_selection import train_test_split




def get_dataset(dataset_name, n_pub_sets = 1) :

    central_test_set, central_train_set = None, None
    if dataset_name == 'depth': 
        train_sets, test_sets = get_depth_data(onehot=True, as_torch_dataset=False) 
        public_set = (np.concatenate([train_sets[i][0] for i in range(n_pub_sets)]), np.concatenate([train_sets[i][1] for i in range(n_pub_sets)]))
        local_sets = [train_sets[i] for i in range(n_pub_sets, len(train_sets))]
        test_sets  = [test_sets[i] for i in range(n_pub_sets, len(test_sets))]
        try : 
            central_train_set = (np.concatenate([train_sets[i][0] for i in range(n_pub_sets, len(local_sets))]), np.concatenate([train_sets[i][1] for i in range(n_pub_sets, len(local_sets))]))
            central_test_set = (np.concatenate([test_sets[i][0] for i in range(n_pub_sets, len(test_sets))]), np.concatenate([test_sets[i][1] for i in range(n_pub_sets, len(test_sets))]))        
        except :
            print("could not concatenate train/test sets")
            print("n_pub_sets = ", n_pub_sets, " len(local_sets) = ", len(local_sets))
        
        return central_train_set, central_test_set, public_set, local_sets, test_sets

    elif dataset_name == 'hars':
        (pri_x_list, pri_y_list), (pri_x_total, pri_y_total), (x_test, y_test_cat), (pub_x, pub_y_cat), original_labels = get_hars_data(n_parties = 9, n_samples_per_class= 200)
        public_set = (pub_x, pub_y_cat)
        local_sets = [(pri_x_list[i], pri_y_list[i]) for i in range(len(pri_x_list))]
        test_sets  = [(x_test, y_test_cat) for i in range(len(pri_x_list))]
        central_train_set = (pri_x_total, pri_y_total)
        central_test_set = (x_test, y_test_cat)
        return central_train_set, central_test_set, public_set, local_sets, test_sets
        
    elif dataset_name == 'harbox':
        train_sets, test_sets = get_harbox_data(data_dir = '../data/large_scale_HARBox', onehot=True, as_torch_dataset=False)
        public_set = (np.concatenate([train_sets[i][0] for i in range(n_pub_sets)]), np.concatenate([train_sets[i][1] for i in range(n_pub_sets)]))
        local_sets = [train_sets[i] for i in range(n_pub_sets, len(train_sets))]
        test_sets  = [test_sets[i] for i in range(n_pub_sets, len(test_sets))]
        try : 
            central_train_set = (np.concatenate([train_sets[i][0] for i in range(n_pub_sets, len(local_sets))]), np.concatenate([train_sets[i][1] for i in range(n_pub_sets, len(local_sets))]))
            central_test_set = (np.concatenate([test_sets[i][0] for i in range(n_pub_sets, len(test_sets))]), np.concatenate([test_sets[i][1] for i in range(n_pub_sets, len(test_sets))]))        
        except :
            print("could not concatenate train/test sets")
            print("n_pub_sets = ", n_pub_sets, " len(local_sets) = ", len(local_sets))
        
        return central_train_set, central_test_set, public_set, local_sets, test_sets
    
    elif dataset_name == 'imu':
        train_sets, test_sets = get_imu_data(data_dir = '../data/imu_dataset', onehot=True, as_torch_dataset=False)
        public_set = (np.concatenate([train_sets[i][0] for i in range(n_pub_sets)]), np.concatenate([train_sets[i][1] for i in range(n_pub_sets)]))
        local_sets = [train_sets[i] for i in range(n_pub_sets, len(train_sets))]
        test_sets  = [test_sets[i] for i in range(n_pub_sets, len(test_sets))]
        try : 
            central_train_set = (np.concatenate([train_sets[i][0] for i in range(n_pub_sets, len(local_sets))]), np.concatenate([train_sets[i][1] for i in range(n_pub_sets, len(local_sets))]))
            central_test_set = (np.concatenate([test_sets[i][0] for i in range(n_pub_sets, len(test_sets))]), np.concatenate([test_sets[i][1] for i in range(n_pub_sets, len(test_sets))]))        
        except :
            print("could not concatenate train/test sets")
            print("n_pub_sets = ", n_pub_sets, " len(local_sets) = ", len(local_sets))
        
        return central_train_set, central_test_set, public_set, local_sets, test_sets

    elif dataset_name == 'cifar100':

        pass

    else:
        raise NotImplementedError










def get_hars_data(n_parties, n_samples_per_class, dataset_dir = None, include_classes = 'all') : 

    if dataset_dir is None : 
        dataset_dir = '../data/HARS'
        
    test_data = pd.read_csv(os.path.join(dataset_dir,'test.csv'))
    train_data = pd.read_csv(os.path.join(dataset_dir,'train.csv'))

    train_data = train_data.sample(frac = 1.0)

    train_data_len = int(0.9 * len(train_data) ) 
    train_data, public_data = train_data.iloc[:train_data_len], train_data.iloc[train_data_len:]

    # Eliminate last two columns from the x data ('subject', 'label') 
    x_train, y_train = train_data.iloc[:, :-2], train_data.iloc[:, -1:] 
    x_test, y_test = test_data.iloc[:, :-2], test_data.iloc[:, -1:]


    pub_x, pub_y = public_data.iloc[:, :-2], public_data.iloc[:, -1:]


    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    pub_y = le.transform(pub_y) 

    scaling_data = MinMaxScaler()
    x_train = scaling_data.fit_transform(x_train)
    x_test = scaling_data.transform(x_test)
    pub_x = scaling_data.transform(pub_x) 

    n_classes = len(le.classes_)
    original_labels = le.classes_

    if n_parties == 0 : 
        return (x_train, y_train), (x_test, y_test)

    pri_x_list, pri_y_list, pri_x_total, pri_y_total  = split_dataset(x_train, y_train, original_labels, include_classes = include_classes, samples_per_class = n_samples_per_class ,\
                                                                    n_models = n_parties, to_categorical = True) 
    y_train_cat = to_categorical(y_train, num_classes = n_classes)
    y_test_cat = to_categorical(y_test, num_classes = n_classes)
    pub_y_cat = to_categorical(pub_y, num_classes = n_classes)

    return (pri_x_list, pri_y_list), (pri_x_total, pri_y_total), (x_test, y_test_cat), (pub_x, pub_y_cat), original_labels 


def get_hars_dataloader(data_dir, batchsize): 
    (X_train, y_train), (X_test, y_test) = get_hars_data(data_dir, n_parties = 0, n_samples_per_class = 0, include_classes = 'all')
    y_train = tf.keras.utils.to_categorical(y_train, num_classes = 6)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes = 6)
    trainloader = torch.utils.data.DataLoader(torch_data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=batchsize, shuffle=True)
    testloader = torch.utils.data.DataLoader(torch_data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)), batch_size=batchsize, shuffle=False)
    return trainloader, testloader


def get_depth_data(onehot = True, as_torch_dataset = False): 
    
    temp_n_parties = 9
    x_train, y_train, x_test, y_test = [], [], [], []
    for user_id in range(temp_n_parties):
        x_train_i, y_train_i = depth_pre.load_depth_data_train(user_id)
        # add channel dimension
        x_train_i = np.expand_dims(x_train_i, axis = -1) 
        x_train.append(x_train_i)
        y_train.append(y_train_i)

    for user_id in range(temp_n_parties):
        x_test_i, y_test_i = depth_pre.load_depth_data_test(user_id)
        # add channel dimension
        x_test_i = np.expand_dims(x_test_i, axis = -1)
        x_test.append(x_test_i)
        y_test.append(y_test_i)
    
    if onehot :
        y_train, y_test = [to_categorical(y_train[i], num_classes = 5) for i in range(9)], [to_categorical(y_test[i], num_classes = 5) for i in range(9)]
 
    if as_torch_dataset :
        return [HARDataset(x_train[i], y_train[i]) for i in range(temp_n_parties)], [HARDataset(x_test[i], y_test[i]) for i in range(temp_n_parties)]
    else :
        return [(x_train[i], y_train[i]) for i in range(temp_n_parties)], [(x_test[i], y_test[i]) for i in range(temp_n_parties)]

    
def get_harbox_data(data_dir, onehot=True, as_torch_dataset=False): 

    x_train, y_train, x_test, y_test = [], [], [], []
    for user_id in range(1, harbox_pre.NUM_OF_TOTAL_USERS + 1):
        x_train_i, y_train_i, _, _ = harbox_pre.load_data(data_dir, user_id)
        x_train_i, x_test_i, y_train_i, y_test_i = train_test_split(x_train_i, y_train_i, test_size=0.3, random_state=42)
        x_train.append(x_train_i)
        y_train.append(y_train_i)
        x_test.append(x_test_i)
        y_test.append(y_test_i)

    if onehot :
        y_train, y_test = [to_categorical(y_train[i], num_classes = harbox_pre.NUM_OF_CLASS) for i in range(harbox_pre.NUM_OF_TOTAL_USERS)], [to_categorical(y_test[i], num_classes = harbox_pre.NUM_OF_CLASS) for i in range(harbox_pre.NUM_OF_TOTAL_USERS)]

    if as_torch_dataset :
        return [HARDataset(x_train[i], y_train[i]) for i in range(harbox_pre.NUM_OF_TOTAL_USERS)], [HARDataset(x_test[i], y_test[i]) for i in range(harbox_pre.NUM_OF_TOTAL_USERS)]
    else :
        return [(x_train[i], y_train[i]) for i in range(harbox_pre.NUM_OF_TOTAL_USERS)], [(x_test[i], y_test[i]) for i in range(harbox_pre.NUM_OF_TOTAL_USERS)]


    
def get_imu_data(data_dir, onehot=True, as_torch_dataset=False): 

    x_train, y_train, x_test, y_test = [], [], [], []
    for user_id in range(imu_pre.NUM_OF_TOTAL_USERS):
        x_train_i, y_train_i, _ = imu_pre.load_data(data_dir, user_id)
        x_train_i, x_test_i, y_train_i, y_test_i = train_test_split(x_train_i, y_train_i, test_size=0.3, random_state=42)
        x_train.append(x_train_i)
        y_train.append(y_train_i)
        x_test.append(x_test_i)
        y_test.append(y_test_i)

    if onehot :
        y_train, y_test = [to_categorical(y_train[i], num_classes = imu_pre.NUM_OF_CLASS) for i in range(imu_pre.NUM_OF_TOTAL_USERS)], [to_categorical(y_test[i], num_classes = imu_pre.NUM_OF_CLASS) for i in range(imu_pre.NUM_OF_TOTAL_USERS)]

    if as_torch_dataset :
        return [HARDataset(x_train[i], y_train[i]) for i in range(imu_pre.NUM_OF_TOTAL_USERS)], [HARDataset(x_test[i], y_test[i]) for i in range(imu_pre.NUM_OF_TOTAL_USERS)]
    else :
        return [(x_train[i], y_train[i]) for i in range(imu_pre.NUM_OF_TOTAL_USERS)], [(x_test[i], y_test[i]) for i in range(imu_pre.NUM_OF_TOTAL_USERS)]





def get_dataloader_from_numpy(x, y, batch_size = 32, shuffle = True) : 
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    dataset = torch_data.TensorDataset(x, y)
    dataloader = torch_data.DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)
    return dataloader













def split_dataset(x, y, original_labels, samples_per_class, n_models, include_classes, to_categorical = False) : 
    datasets = [None]*n_models 
    labels = [None]*n_models 

    sample_indecies = [None]*n_models 
    n_classes = len(original_labels)
    combined_idx = np.array([], dtype = np.int16) 
     
    all_classes = list(np.arange(n_classes))

    for label in all_classes :
        idx = np.where(y == label)[0]
        idx = np.random.choice(idx, samples_per_class*n_models, replace = True) 
        combined_idx = np.r_[combined_idx, idx]
        for i in range(n_models) :
            if include_classes != 'all' : 
                if label not in include_classes[i] :
                    continue
            if datasets[i] is None :
                datasets[i] = [x[idx[i*samples_per_class : (i+1)*samples_per_class]]]
                labels[i] = [y[idx[i*samples_per_class : (i+1)*samples_per_class]]]
            else : 
                datasets[i].append( x[idx[i*samples_per_class : (i+1)*samples_per_class]])
                labels[i].append(y[idx[i*samples_per_class : (i+1)*samples_per_class]])
  
    for i in range(n_models) : 
        datasets[i] = np.concatenate(datasets[i])
        labels[i] = np.concatenate(labels[i])
    

    total_datasets = x[combined_idx]
    total_labels = y[combined_idx]

    
    if to_categorical : 
        for i, l in enumerate(labels): 
            labels[i] = tf.keras.utils.to_categorical(l, num_classes = n_classes)       
        total_labels = tf.keras.utils.to_categorical(total_labels, num_classes = n_classes)
    
    return datasets, labels, total_datasets, total_labels 






#________________________________________________________________________



def partition_data(dataset, datadir, partition, n_parties, beta=0.4):
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
    elif dataset == 'tinyimagenet':
        X_train, y_train, X_test, y_test = load_tinyimagenet_data(datadir)

    n_train = y_train.shape[0]

    if partition == "homo" or partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}


    elif partition == "noniid-labeldir" or partition == "noniid":
        min_size = 0
        min_require_size = 10
        K = 10
        if dataset == 'cifar100':
            K = 100
        elif dataset == 'tinyimagenet':
            K = 200
            # min_require_size = 100

        N = y_train.shape[0]
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # if K == 2 and n_parties <= 10:
                #     if np.min(proportions) < 200:
                #         min_size = 0
                #         break

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]


    return (X_train, y_train, X_test, y_test, net_dataidx_map)



def get_cifar10_dataloader(datadir, batchsize) : 
    (X_train, y_train, X_test, y_test) = load_cifar10_data(datadir)
    trainloader = torch.utils.data.DataLoader(torch_data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=batchsize, shuffle=True)
    testloader = torch.utils.data.DataLoader(torch_data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)), batch_size=batchsize, shuffle=False)
    return trainloader, testloader

def load_cifar10_data(datadir, fraction = 1.0):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target
    
    # permute data 
    sample_size = (fraction*len(X_train))
    idx_train = np.random.permutation(len(X_train))
    idx_test = np.random.permutation(len(X_test))

    X_train = X_train[idx_train[:int(sample_size)]]
    y_train = y_train[idx_train[:int(sample_size)]]
    X_test = X_test[idx_test[:int(sample_size)]]
    y_test = y_test[idx_test[:int(sample_size)]]


    return (X_train, y_train, X_test, y_test)



class CIFAR10_truncated(torch_data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
            else:
                data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        # img = Image.fromarray(img)
        # print("cifar10 img:", img)
        # print("cifar10 target:", target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)



class CIFAR10_truncated(Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
            else:
                data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        # img = Image.fromarray(img)
        # print("cifar10 img:", img)
        # print("cifar10 target:", target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class CIFAR100_truncated(Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR100(self.root, self.train, self.transform, self.target_transform, self.download)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
            else:
                data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        img = Image.fromarray(img)
        # print("cifar10 img:", img)
        # print("cifar10 target:", target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)




class HARDataset(Dataset):
  def __init__(self, data, labels, transform=None, target_transform=None):
    self.data = data
    self.labels = labels
    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    idx = idx % len(self.data)
    data = self.data[idx].astype(np.float32)
    label = self.labels[idx].astype(np.float32)
    if self.transform:
        data = self.transform(data)
    if self.target_transform:
        label = self.target_transform(label)
    return data, label



if __name__ == "__main__" : 
    dataset = 'depth' 
    central_train_set, central_test_set, public_set, local_sets, test_sets = get_dataset(dataset)
    print(central_train_set[0].shape)