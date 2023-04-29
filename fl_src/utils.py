


# Stopped at 
# Starting FL experiment with imu dataset, C = 1.0, aggregation method = soft_labels, hyperparameter tuning = False, augment = True, weighting = performance_based, n_pub_sets = 1, private = True



import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
# import seaborn as sns
from keras.models import Sequential
# from tensorflow.keras.layers import Dense,Dropout, MaxPooling1D
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
import copy

import tensorflow as tf 


import torch 
import torch.nn as nn
import torch.nn.functional as F
# import transforms

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms


import opacus
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager
import torch.utils.data as torch_data

from tqdm  import tqdm

# # tensorflow-privacy 
# import tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.membership_inference_attack as mia
# from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackInputData
# from tensorflow_privacy.privacy.membership_inference_attack.data_structures import SlicingSpec
# from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackType
# import tensorflow_privacy.privacy.membership_inference_attack.plotting as plotting


from data_utils import get_depth_data, HARDataset, get_hars_data
from model_utils import * 
import itertools 
# ______________________________________________________________________
# ______________________________________________________________________
# ______________________________________________________________________


RESULTS_DIR = "/results"
MODELS_DIR = "/models"

depth_dataset_info = {
    'dir': '/data/depth',
    'n_parties': 9, 

}
hars_dataset_info = {
    'dir': '/data/hars',
    'n_parties': 0, 
    'n_classes': 6,
}
datasets_info = {
    'depth': depth_dataset_info,
    'hars': hars_dataset_info,
}



title_font = {'fontname':'Arial', 'size':'28', 'color':'black', 'weight':'normal'} # Bottom vertical alignment for more space
axis_font = {'fontname':'Arial', 'size':'25'}
tick_font = {'fontname':'Arial', 'size':'23'}
legend_font = {'size':'17'}


# n_parties = 10

n_alignment =  200
n_iterations = 100 









# ______________________________________________________________________
# ______________________________________________________________________
# ______________________________________________________________________


# a function that calculates model size in MB
def get_model_size(state_dict, size = 'KB'):
    scale = 1e3 if size == 'KB' else 1e6
    torch.save(state_dict, "temp.p")
    size = os.path.getsize("temp.p")/scale 
    os.remove('temp.p')
    return size

# a function that calculates numpy array size in MB
def get_array_size(array, size = 'KB'):
    scale = 1e3 if size == 'KB' else 1e6
    size = array.nbytes / scale
    return size

def size_of(obj, size = 'KB') : 
    scale = 1e3 if size == 'KB' else 1e6
    # if obj is a numpy array
    if isinstance(obj, np.ndarray) :
        return get_array_size(obj, size)
    else: 
        return get_model_size(obj, size)
    

        


def aggregate_soft_labels(carrier_labels, target_performance, weighted_averaging = False) : 
    
    if weighted_averaging : 
        aggregate_training_metadata = np.average(carrier_labels, weights = target_performance, axis = 0)
    else : 
        aggregate_training_metadata = np.average(carrier_labels, axis = 0) 

    return aggregate_training_metadata


def collect_metadatas(nodes, seed, alpha) : 
    # Collect training metadata
    pub_scores, target_performances = [], []
    for i, node in enumerate(nodes) : 
        training_metadata, target_performance = node.get_training_metadata(seed, alpha) 
        pub_scores.append(training_metadata)
        target_performances.append(target_performance) 
    return pub_scores, target_performances 


def collect_and_aggregate(nodes, seed, alpha, weighted_averaging = False) :
    pub_scores, target_performances = collect_metadatas(nodes, seed, alpha)
    aggregate_training_metadata = aggregate_soft_labels(pub_scores, target_performances, weighted_averaging)
    return aggregate_training_metadata


class FedAMDNode : 

    def __init__(self, model_id, model, local_dl, public_set, test_dl) : 
        self.model_id = model_id
        self.model = model 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.local_dl = local_dl
        self.public_set = public_set 
        self.test_dl = test_dl

        self.test_accs, self.test_losses = [], []
        self.train_accs, self.train_losses = [], []
        self.cce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.nllloss = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)


    def get_training_metadata(self, seed, alpha) : 
        self.seed = seed 
        self.alpha = alpha
        soft_labels = self.get_soft_labels() 
        performance = self.evaluate_on_test_set()
        return soft_labels, performance


    def get_soft_labels(self) : 
        
        x = self.public_set[0]
        np.random.seed(self.seed) 
        index = np.random.permutation(len(x))  
        mixed_x = self.alpha * x + (1 - self.alpha) * x[index, ...]
        mixed_x_torch = torch.from_numpy(mixed_x).float().to(self.device)
        
        return self.model(mixed_x_torch)[0].detach().cpu().numpy()


    def receive_training_metadata(self, metadata) : 
        self.public_set = (self.public_set[0], metadata )


    def evaluate_on_test_set(self): 
        accs, losses = [], []
        with torch.no_grad():
            for x, y in self.test_dl:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                probs = softmax_with_temperature(logits, 1.0) 
                log_probs = torch.log(probs)
                preds = probs.argmax(-1)
                loss = self.nllloss(log_probs, y.argmax(-1))
                n_correct = float(preds.eq(y.argmax(-1)).sum())
                accs.append( n_correct / len(y))
                losses.append( float(loss) )

        self.test_accs.append(np.mean(accs))
        self.test_losses.append(np.mean(losses))

        return self.test_accs[-1]


    def train_on_target(self, epochs = 3, evaluate = True) :
        
        for epoch in range(epochs) : 
            accs, losses = [], []
            for x, y in self.local_dl:
                x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(x)
                probs = softmax_with_temperature(logits, 1.0)
                log_probs = torch.log(probs)
                preds = probs.argmax(-1)
                
                loss = self.nllloss(log_probs, y.argmax(-1))
                loss.backward()

                self.optimizer.step()

                preds = probs.argmax(-1)
                n_correct = float(preds.eq(y.argmax(-1)).sum())
                accs.append( n_correct / len(y))
                losses.append( float(loss) )
                
            if evaluate : 
                acc = self.evaluate_on_test_set()
                self.test_accs.append(acc) 
                self.test_losses.append(np.mean(losses))

    def train_on_public(self, epochs = 1, evaluate = True) : 
        r_public_dl = DataLoader(HARDataset(self.public_set[0], self.public_set[1]), batch_size = 32, shuffle = False)
        np.random.seed(self.seed)
        for epoch in range(epochs) : 
            accs, losses = [], []
            for x, y in r_public_dl:
                x = x.to(self.device)
                y = y.to(self.device)

                mixed_x = self.alpha * x + (1 - self.alpha) * x[np.random.permutation(len(x)), ...]
                logits, probs = self.model(mixed_x) 
                
                loss = self.mse(logits, y)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

            if evaluate : 
                acc = self.evaluate_on_test_set()
                # print(f"Public training :: Epoch {epoch} - Test acc: {acc}")
                return acc 
                

    def save_model(self) : 
        path = MODELS_DIR + str(self.model_id) + '.pt'
        torch.save(self.model.state_dict(), path)


def run_mia_attack(model, train_dl, test_dl) : 
  """
  given a model, and one-hot encoded train and test labels, the function applies the Membership Inference Attack (MIA) and returns results 
  """
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model.to(device) 
  model.eval()
  cce = nn.CrossEntropyLoss(reduction = 'none')
  train_losses, train_logits, all_train_labels = [], [], []
  for batch in train_dl : 
    train_data, train_labels = batch 
    train_data = train_data.to(device).to(torch.float32)
    train_labels = train_labels.to(device).to(torch.float32)

    logits, probs = model(train_data)
    train_loss = cce(logits, train_labels).detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    train_labels = train_labels.detach().cpu().numpy().astype(int)
    train_losses.append(train_loss)
    train_logits.append(logits)
    all_train_labels.append(np.argmax(train_labels, axis = 1))
  
  train_logits = np.concatenate(train_logits)
  train_losses = np.concatenate(train_losses)
  
  test_losses, test_logits, all_test_labels = [], [], []
  for batch in test_dl :
    test_data, test_labels = batch 
    test_data = test_data.to(device).to(torch.float32)
    test_labels = test_labels.to(device).to(torch.float32)

    logits, probs = model(test_data)
    test_loss = cce(logits, test_labels).detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    test_labels = test_labels.detach().cpu().numpy().astype(int)
    test_losses.append(test_loss)
    test_logits.append(logits)
    all_test_labels.append(np.argmax(test_labels, axis = 1))
    
  test_losses, test_logits = np.concatenate(test_losses), np.concatenate(test_logits)
  all_train_labels, all_test_labels = np.concatenate(all_train_labels), np.concatenate(all_test_labels)
  print(all_train_labels.shape, all_test_labels.shape)
  # define what variables our attacker should have access to
  attack_input = AttackInputData(
  logits_train = train_logits,
  logits_test = test_logits, 
  loss_train = train_losses, 
  loss_test = test_losses, 
  labels_train = all_train_labels, 
  labels_test = all_test_labels
  )
  # define the type of attacker model that we want to use
  attack_types = [
      AttackType.THRESHOLD_ATTACK,
      AttackType.LOGISTIC_REGRESSION, 
      AttackType.MULTI_LAYERED_PERCEPTRON
  ]
  # how should the data be sliced
  slicing_spec = SlicingSpec(
    entire_dataset = True,
    by_class = True,
    by_percentiles = False,
    by_classification_correctness = True)
  
  attacks_result = mia.run_attacks(attack_input=attack_input,
                                 slicing_spec=slicing_spec,
                                 attack_types=attack_types)
  return attacks_result 


def get_heterogeneous_model(client_id, dataset_shape, n_classes) : 
    
    model_id = client_id % 3
    if len(dataset_shape) == 3 : 
        print("Using HAR_TS_Net")
        n_lstm_layers = model_id + 1
        model = HAR_TS_Net(n_lstm_layers = n_lstm_layers, n_features = dataset_shape[-1], n_classes = n_classes)
    elif len(dataset_shape) == 2  :
        if model_id == 0 : 
            model = OneLayerMLP(dataset_shape[1], n_classes = n_classes)
        elif model_id == 1 :
            model = TwoLayerMLP(dataset_shape[1], n_classes = n_classes)
        else :
            model = ThreeLayerMLP(dataset_shape[1], n_classes = n_classes)
    elif len(dataset_shape) == 4 : 
        if model_id == 0 : 
            model = Net_CIFAR()
            # model = HAR_CV_Net(input_shape = dataset_shape[1:], f1 = 32, f2 = 64, f3 = 128, n_classes = n_classes)
        elif model_id == 1 :
            model = HAR_CV_Net(input_shape = dataset_shape[1:], f1 = 4, f2 = 12, f3 = 25, n_classes = n_classes)
        else :
            model = HAR_CV_Net(input_shape = dataset_shape[1:], f1 = 10, f2 = 20, f3 = 25, n_classes = n_classes)
    else : 
        raise ValueError("Dataset shape not supported")
    
    return model


# a function that takes an array of soft labels and returns a compressed version of the array from float32 to int8
def compress_labels(preds) :
        
    preds = preds - preds.min(axis = 1, keepdims = True)
    preds = preds / preds.max(axis = 1, keepdims = True)
    
    preds = preds * 255
    preds = preds.astype(np.uint8)
    
    return preds




class FLClient(nn.Module) : 

    def __init__(self, client_id, data, params) : 
        super(FLClient, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params = params
        self.set_params() 

        self.client_id = client_id
        self.local_set, self.public_set, self.test_set = data
        self.model = get_heterogeneous_model(self.client_id, self.local_set[0].shape, n_classes = self.local_set[1].shape[-1])

        self.train_dataset = torch_data.TensorDataset(torch.tensor(self.local_set[0], dtype=torch.float32).permute(0, 3, 1, 2), torch.tensor(self.local_set[1], dtype=torch.float32))
        self.test_dataset = torch_data.TensorDataset(torch.tensor(self.test_set[0], dtype=torch.float32).permute(0, 3, 1, 2), torch.tensor(self.test_set[1], dtype=torch.float32))
        self.public_dataset = torch_data.TensorDataset(torch.tensor(self.public_set[0], dtype=torch.float32).permute(0, 3, 1, 2), torch.tensor(self.public_set[1], dtype=torch.float32))

        # my_transforms = transforms.Compose(
        #     [transforms.RandomHorizontalFlip(),
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # self.train_dataset = HARDataset(self.local_set[0], self.local_set[1], transform = my_transforms)
        # self.test_dataset = HARDataset(self.test_set[0], self.test_set[1], transform = my_transforms) 
        # self.public_dataset = HARDataset(self.public_set[0], self.public_set[1], transform = my_transforms) 

        self.local_dl = DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True)
        self.test_dl = DataLoader(self.test_dataset, batch_size = self.batch_size, shuffle = True)
        self.public_dl = DataLoader(self.public_dataset, batch_size = self.batch_size, shuffle = True)

        self.local_accs, self.local_losses = [], []
        self.test_accs, self.test_losses = [], []
        self.train_accs, self.train_losses = [], []
        self.public_accs, self.public_losses = [], []

        self.cce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.nllloss = nn.NLLLoss()

        
        if self.params['private'] : 
            # The optimizer will be set in the make_private function

            self.model = ModuleValidator.fix(self.model)
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.params['lr'])
            self.privacy_engine = PrivacyEngine()
            self.model, self.optimizer, self.local_dl = self.privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.local_dl,
                epochs=self.params["epochs"] * self.params['tot_T'],
                target_epsilon=self.epsilon,
                target_delta=self.delta,
                max_grad_norm=self.max_grad_norm,
            )

        else : 
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.params['lr'])

        
        # for i in range(10) : 
        #     train(self.model, self.local_dl, self.cce, self.optimizer, self.privacy_engine, self.delta, None, True) 
        
        
        # self.scheduler = StepLR(self.optimizer, step_size=15, gamma=0.1)
        # self.KD_scheduler = StepLR(self.KD_optimizer, step_size=25, gamma=0.1)

        # if self.params['hyperparameter_tuning'] : 
        #     self.hyperparameter_tuning() 
        #     self.optimizer = optim.SGD(self.model.parameters(), lr=self.best_lr) 
        #     self.local_dl = DataLoader(train_dataset, batch_size = self.best_bs, shuffle = True)
        

    def local_benchmark(self, save_results = True) :
        self.local_model = get_heterogeneous_model(self.client_id, self.local_set[0].shape, n_classes = self.local_set[1].shape[-1])
        self.local_optimizer = optim.SGD(self.local_model.parameters(), lr=self.best_lr)

        for epoch in range(self.params['local_benchmark_epochs']) : 
            _, _ = train(self.local_model, self.local_dl, self.local_optimizer, None, None, None, False)
            test_acc, test_loss = test(self.local_model, self.test_dl, None, None, None, False)
            if save_results: 
                self.local_accs.append(test_acc)
                self.local_losses.append(test_loss)
            
        return test_acc 



    def align_public_set(self, epochs) : 

        
        public_optimizer = optim.SGD(self.model.parameters(), lr=self.params['lr'])
        for epoch in range(epochs) : 
            pub_acc, pub_loss = train(self.model, self.public_dl, public_optimizer, None, None, None, False)
            
            self.public_accs.append(pub_acc)
            self.public_losses.append(pub_loss)
        
    def hyperparameter_tuning(self) :
        print("Client {} is tuning hyperparameters".format(self.client_id), end = " ")
        lr = self.params['lr']
        bs = self.params['batch_size']

        lr_range = [lr / 10, lr, lr * 10]
        bs_range = [int(np.power(2, np.log2(bs)-1)), bs, int(np.power(2, np.log2(bs)+1))]

        best_acc = 0
        best_lr = lr
        best_bs = bs

        for lr in lr_range :
            for bs in bs_range :
                model = get_heterogeneous_model(self.client_id, self.local_set[0].shape, n_classes = self.local_set[1].shape[-1])
                optimizer = optim.SGD(model.parameters(), lr=lr)
                
                dl = DataLoader(self.local_dl.dataset, batch_size = bs, shuffle = True)
                for epoch in range(25) : 
                    _, _ = train(model, dl, self.cce, optimizer, None, None, None, False)
                    test_acc, test_loss = test(model, self.test_dl, self.cce, None, None, None, False)
                    if test_acc > best_acc : 
                        best_acc = test_acc
                        best_lr = lr
                        best_bs = bs
        
        self.best_bs = best_bs
        self.best_lr = best_lr
        print("Best lr: {}, best bs: {}".format(best_lr, best_bs))

    def set_params(self) :
        self.epochs = self.params["epochs"]
        self.max_grad_norm = self.params["max_grad_norm"]
        self.delta = self.params["delta"]
        self.epsilon = self.params["epsilon"]
        self.client_path = self.params["client_path"]
        self.model_path = os.path.join(self.client_path, "model.pt")
        self.batch_size = self.params["batch_size"]
        self.best_lr = self.params['lr']
        self.best_bs = self.params['batch_size']



    def recv_params(self, metadata, compressed = False) : 
        if compressed :
            metadata = metadata.astype(np.float32)
            metadata /= 255

        if self.params['aggregate'] == 'weights' : 
            self.model.load_state_dict(copy.deepcopy(metadata)) 
        else : 
            self.public_set = (self.public_set[0], metadata )


    def communicate_meta(self, beta, lambdaa, augment) : 
    
        self.lambdaa = lambdaa 
        self.beta = beta
        self.augment = augment

        np.random.seed(self.beta) 
        x, y = self.public_set
        index = np.random.permutation(len(x))  

        
        mixed_x = self.lambdaa * x + (1 - self.lambdaa) * x[index]
        mixed_y = self.lambdaa * y + (1 - self.lambdaa) * y[index]

        self.public_set = (mixed_x, mixed_y)

        return None 
    

    def get_soft_labels(self, normalize = False, compress = False) : 
        self.model.eval()
        
        x_torch = torch.from_numpy(self.public_set[0]).permute(0, 3, 1, 2).float().to(self.device)
        preds = self.model(x_torch)
        # soft_labels = softmax_with_temperature(preds, self.params['temperature'])
        preds = preds.detach().cpu().numpy()

        if normalize :
            preds = preds - preds.min(axis = -1, keepdims = True)
            preds = preds / preds.max(axis = -1, keepdims = True)

        if compress :
            # print("compressed from ", size_of(preds), end = "") 
            assert normalize, "Cannot compress if not normalized"
            preds = preds * 255
            preds = preds.astype(np.uint8)
            # print(" to ", size_of(preds))
        
        self.model.train() 
        
        return preds 


    def evaluate_on_test_set(self): 
        accs, losses = [], []
        with torch.no_grad():
            for x, y in self.test_dl:
                x = x.to(self.device).to(torch.float32)
                y = y.to(self.device).to(torch.float32)

                logits = self.model(x)
                probs = F.softmax(logits, dim=-1)
                log_probs = torch.log(probs)
                loss = self.nllloss(log_probs, y.argmax(-1))
                preds = probs.argmax(-1)
                n_correct = float(preds.eq(y.argmax(-1)).sum())
                accs.append( n_correct / len(y))
                losses.append( float(loss) )

        self.test_accs.append(np.mean(accs))
        self.test_losses.append(np.mean(losses))

        return self.test_accs[-1]


    def get_test_acc(self) : 
        return self.test_accs[-1]

    def get_train_acc(self) : 
        return self.train_accs[-1]

    def local_update(self, evaluate = True) :

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        for epoch in range(self.epochs) : 
            accs, losses = [], []
            if self.params['private'] : 
                with BatchMemoryManager(data_loader=self.local_dl, max_physical_batch_size=1, optimizer=self.optimizer) as new_data_loader:  
                    for x, y in new_data_loader:
                        x, y = x.to(torch.float32).to(device), y.to(torch.float32).to(device)
                        
                        logits = self.model(x)
                        probs = F.softmax(logits, dim=-1)
                        log_probs = torch.log(probs)
                        loss = self.nllloss(log_probs, y.argmax(-1))
                        loss.backward()

                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        # self.scheduler.step()

                        preds = probs.argmax(-1)
                        n_correct = float(preds.eq(y.argmax(-1)).sum())
                        accs.append( n_correct / len(y))
                        losses.append( float(loss) )
            else : 
                for x, y in self.local_dl:
                    x, y = x.to(torch.float32).to(device), y.to(torch.float32).to(device)

                    logits = self.model(x)
                    probs = F.softmax(logits, dim=-1)
                    log_probs = torch.log(probs)
                    loss = self.nllloss(log_probs, y.argmax(-1))
                    loss.backward()

                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # self.scheduler.step()

                    preds = probs.argmax(-1)
                    n_correct = float(preds.eq(y.argmax(-1)).sum())
                    accs.append( n_correct / len(y))
                    losses.append( float(loss) )
            
            if evaluate : 
                acc = self.evaluate_on_test_set()
                self.test_accs.append(acc) 
                self.train_accs.append(np.mean(accs))
                self.test_losses.append(np.mean(losses))



    def digest(self) : 
        
        # self.model.attribute = list(self.model.attribute)  # where attribute was dict_keys
        model_clone = get_heterogeneous_model(self.client_id, self.local_set[0].shape, n_classes = self.local_set[1].shape[-1])
        model_params = get_state_dict_copy(self.model)
        # rename model_parma keys to match model_clone keys
        
        model_params_renamed = {k.replace('_module.', ''): v for k, v in model_params.items()}
        

        model_clone.load_state_dict(model_params_renamed)
        KD_optimizer = optim.SGD(model_clone.parameters(), lr=self.params['lr'])

        
        public_dataset = torch_data.TensorDataset(torch.tensor(self.public_set[0], dtype=torch.float32).permute(0, 3, 1, 2), torch.tensor(self.public_set[1], dtype=torch.float32))
        self.public_dl = DataLoader(public_dataset, batch_size = 32, shuffle = True)
        # self.model.train()

        KD_epochs = max(2, self.epochs // 2)
        for epoch in range(KD_epochs) :
            
            for x, y in self.public_dl : 
                x = x.to(self.device).to(torch.float32)
                y = y.to(self.device).to(torch.float32)

                logits = model_clone(x)
                # soft_labels = softmax_with_temperature(logits, self.params['temperature'])
                loss = self.mse(logits, y)
                loss.backward()

                KD_optimizer.step()
                KD_optimizer.zero_grad()
                    # self.KD_scheduler.step()

        
        prefix = "_module." if self.params['private'] else ""
        model_clone_params_renamed = { prefix+k: v for k, v in copy.deepcopy(model_clone.state_dict()).items()}
        
        self.model.load_state_dict(model_clone_params_renamed)

    def save_assets(self) : 
        # self.model.cpu()
        # torch.save(self.model.state_dict(), self.model_path)
        
        local_df = {
            'local_acc' : self.local_accs, 
            'local_loss' : self.local_losses,
        }
        fl_df = {
            'fl_acc' : self.test_accs,
            'fl_loss' : self.test_losses

        }
        local_df = pd.DataFrame(local_df)
        fl_df = pd.DataFrame(fl_df)
        local_log_path = os.path.join(self.client_path, "local_log.csv")
        fl_log_path = os.path.join(self.client_path, "fl_log.csv")
        local_df.to_csv(local_log_path)
        fl_df.to_csv(fl_log_path)







class Aggregator () : 
    def __init__(self, aggregation_method) : 
        self.aggregation_method = aggregation_method
        
    
    def aggregate(self, clients, idxs_users, weights) :
        if self.aggregation_method == "weights" : 
            return self.aggregate_weights(clients, idxs_users)
        elif self.aggregation_method == "soft_labels" :
            return self.aggregate_soft_labels(clients, idxs_users, weights)
        else : 
            raise NotImplementedError


    @staticmethod
    def aggregate_soft_labels(clients, idxs_users, weights = None, compress = False) : 
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        all_soft_labels = [clients[idx].get_soft_labels(normalize = compress, compress = compress) for idx in idxs_users]
        # print means of soft labels
        # for idx, soft_labels in enumerate(all_soft_labels):
        #     print("mean of soft labels for client {} : {}".format(idx, np.mean(soft_labels)))

        C = clients[0].params["C"]
        if weights is None : 
            weights = np.ones(len(idxs_users)).astype(np.float32) 
        global_soft_labels = np.zeros(all_soft_labels[0].shape)
        for idx, soft_labels in enumerate(all_soft_labels):
            w = weights[idx] / np.sum(weights)
            # not sure about this, but will be ok if C = 1.0
            global_soft_labels += (soft_labels * (w / C))

            

        return all_soft_labels, global_soft_labels
    
    @staticmethod
    def aggregate_weights(clients, idxs_users, weights = None) : 
    
        """FedAvg"""
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        models_par = [clients[idx].model.state_dict() for idx in idxs_users]
        new_par = {}
        C = clients[0].params["C"]
        for name in models_par[0]:
            new_par[name] = torch.zeros(models_par[0][name].shape).to(device)
        if weights is None : 
            weights = np.ones(len(idxs_users)).astype(np.float32) 
        for idx, par in enumerate(models_par):
            w = weights[idx] / np.sum(weights)
            for name in new_par:
                new_par[name] += par[name].to(device) * (w / C)
        return models_par, new_par 


def get_state_dict_copy(model) : 
    par = model.state_dict()
    par_copy = copy.deepcopy(par)
    return par_copy

# a function that takes a pytorch model and its name and prints the name of each layer and its size, one layer per line
def print_model_params(model, name) : 
    print("Model {} has {} layers".format(name, len(list(model.state_dict().keys()))))
    for idx, key in enumerate(model.state_dict().keys()):
        print("Layer {} : {} has size {}".format(idx, key, model.state_dict()[key].size()))


class FLServer(nn.Module):
    
    def __init__(self, fl_param):
        super(FLServer, self).__init__()
        self.params = fl_param
        self.client_num = fl_param['client_num']
        self.C = fl_param['C']      # (float) C in [0, 1]
        self.T = fl_param['tot_T']  # total number of global iterations (communication rounds)
        self.augment = fl_param['augment']

        # create a result folder if not exist
        if not os.path.exists(fl_param['exp_path']):
            os.makedirs(fl_param['exp_path'])

        
        self.clients = [] 
        for i in range(self.client_num):
            # create a directory for each client
            client_path = os.path.join(fl_param['exp_path'], 'client_{}'.format(i))
            if not os.path.exists(client_path):
                os.makedirs(client_path)
            fl_param['client_path'] = client_path
            client_id = 0 #i if 'soft_labels' in fl_param['aggregate'] else fl_param['default_client_id']
            data = (fl_param['local_sets'][i], fl_param['public_set'], fl_param['test_sets'][i])
            self.clients.append(FLClient(client_id, data, fl_param))
        
        # initial alignment
        if 'soft_labels' in self.params['aggregate'] and self.params['initial_pub_alignment_epochs'] > 0 : 
            for client in self.clients : 
                client.align_public_set(self.params['initial_pub_alignment_epochs'])


    def broadcast(self, new_par):
        """Send aggregated model to all clients"""
        for client in self.clients:
            client.recv_params(new_par.copy(), compressed = 'compress' in self.params['aggregate'])

    def broadcast_meta(self, beta, lambdaa):
        
        for client in self.clients:
            client.communicate_meta(beta, lambdaa, augment = self.augment)

    def save_assets(self ) : 
        for client in self.clients : 
            client.save_assets()

    def global_update(self):
        idxs_users = np.random.choice(range(len(self.clients)), int(self.C * len(self.clients)), replace=False)        

        if self.params['aggregate'] == 'soft_labels':
            
            if self.params['augment'] : 
                alpha = np.random.randint(1, 1_000_000)
                beta = np.random.randint(1, 1000)
                lambdaa = np.random.beta(alpha, alpha)
                self.broadcast_meta(beta = beta, lambdaa=lambdaa)
            if self.params['weighting'] == 'uniform':
                weights = [1 for idx in idxs_users]
            else : weights = [self.clients[idx].get_test_acc() for idx in idxs_users]
            all_soft_labels, global_soft_labels = Aggregator.aggregate_soft_labels(self.clients, idxs_users, weights = weights, compress = False)
            self.broadcast(global_soft_labels)
            for idx in idxs_users:
                self.clients[idx].digest()
        elif self.params['aggregate'] == 'compressed_soft_labels':
            alpha = np.random.randint(1, 1_000_000)
            beta = np.random.randint(1, 1000)
            lambdaa = np.random.beta(alpha, alpha)
            self.broadcast_meta(beta = beta, lambdaa=lambdaa)

            if self.params['weighting'] == 'uniform':
                weights = [1 for idx in idxs_users]
            else : weights = [self.clients[idx].get_test_acc() for idx in idxs_users]
            
            all_soft_labels, global_soft_labels = Aggregator.aggregate_soft_labels(self.clients, idxs_users, weights = weights, compress = True)
            self.broadcast(global_soft_labels)
            for idx in idxs_users:
                self.clients[idx].digest()
        elif self.params['aggregate'] == 'weights':
            if self.params['weighting'] == 'uniform':
                weights = [1 for idx in idxs_users]
            else : weights = [self.clients[idx].get_test_acc() for idx in idxs_users]
            
            all_weights, global_weights = Aggregator.aggregate_weights(self.clients, idxs_users, weights = weights) 
            self.broadcast(global_weights)
        else : 
            raise NotImplementedError
        
        for idx in idxs_users:
            self.clients[idx].local_update()
        

        all_accs = [self.clients[idx].get_test_acc() for idx in idxs_users]
        all_train_accs = [self.clients[idx].get_train_acc() for idx in idxs_users]
        
        avg_train_acc = np.mean(all_train_accs)
        avg_acc = np.mean(all_accs)
        min_acc = np.min(all_accs)
        max_acc = np.max(all_accs)
        
        return avg_acc, min_acc, max_acc, avg_train_acc


class Experiment(): 

    def __init__(self, params) : 
        self.set_params(params)

        if not os.path.exists(self.params['exp_path']):
            print("could not find the experiment path")
            raise FileNotFoundError
        
        self.all_fl_accs = [pd.read_csv(f)['fl_acc'].values for f in [os.path.join(self.dir, f, 'fl_log.csv') for f in os.listdir(self.dir) if f.startswith('client')]]
        self.local_accs = pd.read_csv(os.path.join(self.dir, 'client_0', 'local_log.csv'))['local_acc'].values
        self.last_fl_accs = [pd.read_csv(f)['fl_acc'].values.max() for f in [os.path.join(self.dir, f, 'fl_log.csv') for f in os.listdir(self.dir) if f.startswith('client')]]


    def get_fl_acc(self, avg = True) :
        if avg : return np.mean(self.last_fl_accs)
        else : return self.last_fl_accs
    

    def get_local_acc(self, last = True) :
        if last : return self.local_accs[-1]
        else : return self.local_accs
    

    def set_params(self, params) : 
        
        self.params = params
        self.dir = params['exp_path']
        self.name = params['name']
        self.agg = params['aggregate']
        self.C = params['C']
    




# if __name__ == "__main__" : 
#     results_dir = '../thesis_results'
#     datasets = ['hars', 'imu', 'depth']
#     aggregates = ['soft_labels', 'weights']
#     Cs = [1.0]
#     HT = [True, False] # hyperparameter tuning
#     W = ['uniform', 'performance_based'] # weighting
#     Aug = [True, False] # Augmented Knowledge Transfer
#     DP = [False, True] # Differential Privacy

#     columns = ['aggregate', 'Aug', 'W', *datasets]
#     df = pd.DataFrame(columns = columns)

#     ht = HT[1]
#     dp = DP[1]
#     n_pub_sets = 1

#     local_acc_dict = { dataset : [] for dataset in datasets}
#     for aggregate, C, aug, w in itertools.product(aggregates, Cs, Aug, W) :
#         if aug and not aggregate.endswith('soft_labels') : continue

#         dataset_scores = [] 
#         for dataset in datasets :
#             exp_name = 'DP{}/N_pub{}/{}/Agg{}_C{}_HT{}_Aug{}_W{}'.format(dp, n_pub_sets, dataset, aggregate, C, ht, aug, w)
#             exp_params = {
#                 'name' : exp_name,
#                 'exp_path' : os.path.join(results_dir, exp_name),
#                 'dataset' : dataset,
#                 'aggregate' : aggregate,
#                 'C' : C,
#             }

#             # check if the experiment exists
#             if not os.path.exists(exp_params['exp_path']) :
#                 print("could not find the experiment path")
#                 break
            
#             print(exp_params['exp_path'])
#             exp = Experiment(exp_params)
#             fl_avg = exp.get_fl_acc(avg = True)
#             local_acc = exp.get_local_acc(last = True)
#             local_acc_dict[dataset].append(local_acc)

#             dataset_scores.append(round(fl_avg, 2)) 
        
#         row = [aggregate, aug, w, *dataset_scores]
#         df.loc[len(df)] = row

#     for d in datasets : 
#         print(d) 
#         print(local_acc_dict[d])
#     local_row = [np.mean(local_acc_dict[d]) for d in datasets]
#     df.loc[len(df)] = ['local', 'central', *local_row]



#     # grouby-by Agg and HT
    
#     df.to_csv(os.path.join(results_dir, 'results_DP{}.csv'.format(dp)), index = False)
    

#     print(df) 
