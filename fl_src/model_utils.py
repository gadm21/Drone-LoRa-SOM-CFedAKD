
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from statistics import mean
from tqdm import tqdm




class MyLSTM(nn.Module):
    def __init__(self, n_features, n_hidden, n_classes, return_sequences=False):
        super(MyLSTM, self).__init__()
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_features = n_features
        self.return_sequences = return_sequences

        # lstm1, lstm2, linear are all layers in the network
        self.lstm1 = nn.LSTMCell(n_features, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, n_classes)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()


    def forward(self, y):
        outputs, n_samples = [], y.size(0)
        h_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        h_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        
        for i, time_step in enumerate(y.split(1, dim=1)):
            h_t, c_t = self.lstm1(torch.squeeze(time_step), (h_t, c_t)) # initial hidden and cell states
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2)) # new hidden and cell states
            h_t2 = self.tanh(h_t2)
            output = self.linear(h_t2) # output from the last FC layer
            output = self.relu(output)
            outputs.append(output)
        
        if self.return_sequences:
            outputs = torch.stack(outputs, 1).squeeze(2)
        else:
            outputs = outputs[-1]

        return outputs

class HAR_TS_Net(nn.Module): 

    def __init__(self, n_lstm_layers, n_features, n_classes, bidirectional = True):
        super(HAR_TS_Net, self).__init__()
        self.n_lstm_layers = n_lstm_layers
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_hidden = n_features

        self.lstm = nn.LSTM(n_features, self.n_hidden, self.n_lstm_layers, batch_first=True, bidirectional=bidirectional)
        lin_in = self.n_hidden * 2 if bidirectional else self.n_hidden
        self.linear = nn.Linear(lin_in, n_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x) : 
        x, _ = self.lstm(x)
        # get the last hidden state
        x = x[:, -1, :]
        x = self.relu(x) 
        y = self.linear(x)
        y = self.softmax(y) 

        return y
    

# class HAR_T_Net for tabular data
class HAR_T_Net(nn.Module):
    def __init__(self, n_features,  n_classes):
        super(HAR_T_Net, self).__init__()
        self.n_classes = n_classes
        self.n_features = n_features

        self.linear1 = nn.Linear(n_features, self.n_features // 2)
        self.linear2 = nn.Linear(self.n_features // 2, self.n_features // 3)
        self.linear3 = nn.Linear(self.n_features // 3, n_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        # get two outputs for knowledge distillation, one for classification and one for regression
        y1 = x 
        y2 = self.softmax(x)
        
        return y1, y2

# class HAR_CV_Net for computer vision data
class HAR_CV_Net(nn.Module):
    def __init__(self, input_shape, f1, f2, f3, n_classes):
        super(HAR_CV_Net, self).__init__()
        self.f1, self.f2, self.f3 = f1, f2, f3
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.n_hidden = 256

        self.dropout1 = nn.Dropout2d(0.1)

        self.conv1 = nn.Conv2d(input_shape[-1], self.f1, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(self.f1, self.f2, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(self.f2, self.f3, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(self.f3, self.n_hidden, kernel_size=3, stride=1)
        

        # adaptive maxpooling layer
        # self.adaptive_maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(1024, 128)
        self.linear2 = nn.Linear(128, n_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        
        # if x.shape[1] != 1: add another dimension to make the input shape (batch_size, C, H, W)
        if len(x.shape) == 3 : 
            x = x.unsqueeze(1)
        # if channel is not the first dimension, move it to the first dimension
        if x.shape[1] != 1:
            x = x.permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # x = self.adaptive_maxpool(x)

        x = x.flatten(start_dim=1)
        
        x = self.linear1(x)
        x = self.relu(x)

        x = self.linear2(x)
        
        # get two outputs for knowledge distillation, one for classification and one for regression
        y1 = x 
        y2 = self.softmax(x)
        
        return y1, y2

# remove the last two layer of pytorch model to change the output dimension
# def change_output_dim(model, remove_layers, n_classes):

#     model = nn.Sequential(*list(model.children())[:-1*remove_layers])
#     output_dim = model[-1].out_features
#     model.add_module('linear1', nn.Linear(output_dim, n_classes))
#     model.add_module('softmax', nn.Softmax(dim = 1))
#     return model


class OneLayerMLP(nn.Module):
    """Neural Networks"""
    def __init__(self, input_dim, n_classes):
        super(OneLayerMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, int((1/2)*input_dim)),
            nn.Dropout(0.1),
            nn.ReLU(),
            
            
            nn.Linear(int((1/2)*input_dim), n_classes))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.model(x)
        y2 = self.softmax(y)
        return y, y2


class TwoLayerMLP(nn.Module):
    """Neural Networks"""
    def __init__(self, input_dim, n_classes):
        super(TwoLayerMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, int((1/5)*input_dim)),
            nn.Dropout(0.1),
            nn.ReLU(),
  
            nn.Linear(int((1/5)*input_dim), n_classes))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.model(x)
        y2 = self.softmax(y)
        return y, y2

class ThreeLayerMLP(nn.Module):
    """Neural Networks"""
    def __init__(self, input_dim, n_classes):
        super(ThreeLayerMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, int((1/3)*input_dim)),
            nn.Dropout(0.1),
            nn.ReLU(),
            
            nn.Linear(int((1/3)*input_dim), int((1/10)*input_dim)),
            nn.ReLU(),
            
            nn.Linear(int((1/10)*input_dim),  n_classes))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.model(x)
        y2 = self.softmax(y)
        return y, y2



class TwoLayerCNN(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(TwoLayerCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.fc = nn.Linear(7 * 7 * 32, n_classes)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x2 = self.softmax(x)

        return x, x2



def train(model, train_loader, criterion, optimizer, privacy_engine = None, DELTA = None, device = None, verbose = True):

    if device is None : 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    
    accs = []
    losses = []
    iterator = train_loader
    for x, y in iterator:
        x = x.to(device).to(torch.float32)
        y = y.to(device).to(torch.float32)

        logits, probs = model(x)
        loss = criterion(probs, y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        preds = probs.argmax(-1)
        n_correct = float(preds.eq(y.argmax(-1)).sum())
        batch_accuracy = n_correct / len(y)

        accs.append(batch_accuracy)
        losses.append(float(loss))

    if privacy_engine is not None:
        # epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent()  # Yep, we put a pointer to privacy_engine into your optimizer :)
        epsilon = privacy_engine.get_epsilon(DELTA)
        if verbose : print(f"(ε = {epsilon:.3f}, δ = {DELTA})")
    if verbose : 
        print(
            f"Train Accuracy: {mean(accs):.6f}"
            f"Train Loss: {mean(losses):.6f}"
        ) 
    return mean(accs), mean(losses)


def test(model, test_loader, criterion, privacy_engine = None, DELTA = None, device = None, verbose = True):

    if device is None : 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accs, losses = [], []
    with torch.no_grad():
        iterator = test_loader
        for x, y in iterator : 
            x = x.to(device).to(torch.float32)
            y = y.to(device).to(torch.float32)

            logits, preds = model(x)
            loss = criterion(preds, y) 
            preds = preds.argmax(-1)
            n_correct = float(preds.eq(y.argmax(-1)).sum())
            batch_accuracy = n_correct / len(y)

            accs.append(batch_accuracy)
            losses.append(float(loss))

    if verbose :
        print(f"Test Accuracy: {mean(accs):.6f}")
        print(f"Test Loss: {mean(losses):.6f}")
    if privacy_engine is not None : 
        epsilon = privacy_engine.get_epsilon(DELTA)
        if verbose : print(f"(ε = {epsilon:.2f}, δ = {DELTA})")
        

    return mean(accs), mean(losses)


if __name__ == "__main__" : 
    sample = torch.randn(1, 32, 32, 3)
    model = HAR_CV_Net(input_shape = sample.shape[1:], f1 = 32, f2 = 64, f3 = 128, n_classes = 10)
    newmodel = change_output_dim(model, remove_layers = 3, n_classes = 100) 
    print(newmodel)

    out1, out2 = model(sample)
    # out2 = newmodel(sample)
    print(out1.shape)
    print(out2.shape)