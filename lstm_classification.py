import os
import time
import torch
import joblib
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from baseline_config import (
    CLASS_FEATURE_FORMAT
)

#GPU Check
cuda = torch.cuda.is_available()
if cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

#Global parameters
best_loss = float("inf")

#Load the Data
def load_and_preprocess(
    feature_file: str,
    mode: str = "train",
): 
    #load the data and save in dataframe
    dataframe = pd.read_pickle(feature_file)
    features_data = np.stack(dataframe["FEATURES"].values) #shape: [5,50,8]
    decision_data = np.stack(dataframe["GT"].values) #shape: [5,]
    
    #load the features from dataframe
    input_features_data = features_data.astype(float)
    _input = input_features_data[:,:20] #shape: [5,20,8]
    _output = decision_data.astype(int) #shape: [5,]


    data_dict = {
        "input": _input,
        "output": _output,
    }
    
    return data_dict

class ClassRNN(nn.Module):
    def __init__(self, 
                 input_dim:int = 8, 
                 hidden_dim:int = 32, 
                 layer_dim:int = 3, 
                 output_dim:int = 2):
        super(ClassRNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        # Building LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # just want last time step output 
        out = self.fc(out[:, -1, :])
        return out


def train(train_loader, epoch ,loss_function, rnn, rnn_optimizer,n_layers, batch_size):  
    loss_list = []
    for i, (_input, target) in enumerate(train_loader):

        _input = _input.to(device) #torch.Size([5, 20, 8])
        target = target.to(device) #torch.Size([1])
        
        #set rnn to train mode
        rnn.train()

        #clear up the gradient for optimizer
        rnn_optimizer.zero_grad()

        #Encoder observed trajectory
        predictions = rnn(_input)
        
        #Normalize the loss
        loss = loss_function(predictions, target)
        loss_list.append(loss)
        #Backpropagation
        loss.backward()
        rnn_optimizer.step()

        #print(f"Train -- Epoch:{epoch}, loss:{loss}")
    train_loss = sum(loss_list)/len(loss_list)
    return train_loss


class Dataset_Loader(Dataset):
    """Pytorch map-style dataset"""
    def __init__(self, data_dict, mode):
        """Initialize the Dataset_Loader class

        Args:
        data_dict: Dictionary containing desired data
        mode: train/val/test

        """
        self.data_dict = data_dict
        self.mode = mode

        #Get the data
        self.input_data = data_dict["input"]
        self.output_data = data_dict["output"]
        self.num_samples = self.input_data.shape[0]

    def __len__(self):

        return self.num_samples

    def __getitem__(self, idx):

        return (
            torch.FloatTensor(self.input_data[idx]),
            self.output_data[idx]
        )
   

def main():
    #Directories
    train_dir = 'features/classfeatuer/features_class.pkl'
    val_dir = 'features/classfeatuer/features_class.pkl'
    test_dir = "data/test_obs/data"
    
    #Hyperparameters
    batch_size = 4
    lr = 0.001
    num_epochs = 10
    epoch = 0
    input_dim = 8
    hidden_dim = 32
    layer_dim = 3
    output_dim = 2

    print(f"Using all ({joblib.cpu_count()}) CPUs....")
    if cuda:
        print(f"Using all ({torch.cuda.device_count()}) GPUs...")

    #Get the data, in dictionary format
    data_dict = load_and_preprocess(train_dir)
    val_dict = load_and_preprocess(val_dir)
    data_input = data_dict["input"]
    data_output = data_dict["output"]

    #Get the model
    rnn = ClassRNN(input_dim, hidden_dim, layer_dim, output_dim)
    rnn.to(device)

    loss_function = nn.CrossEntropyLoss()
    rnn_optimizer = Adam(rnn.parameters(), lr = lr)

    #Training
    #Transform data to Pytorch Dataset format
    train_dataset = Dataset_Loader(data_dict, "train")
    val_dataset = Dataset_Loader(val_dict, "val")
    
    # print(train_dataset[0][0].shape) #shape: torch.size([20, 8])
    # print(train_dataset[0][1].shape) #shape: torch.size([1])

    #Setting Dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        drop_last = False,
        shuffle = False,
    )

    print("Training begins ...")
    
    global_start_time = time.time()
    best_loss = float("inf")
    prev_loss = best_loss
    while epoch < num_epochs:
        start = time.time()
        loss = train(
            train_loader,
            epoch,
            loss_function,
            rnn,
            rnn_optimizer,
            layer_dim,
            batch_size
        )
        end = time.time()

        if epoch % 1==0:
            print(
                f"Epoch: {epoch+1} completed in: {end-start}s, Total time: {(end - global_start_time) /60.0} mins, loss is: {loss}"
            )

        epoch+=1

    #     if epoch % 500==0:
    #         val_start_time = time.time()
    #         model_loss, avg_dis = validate(
    #             val_loader,
    #             epoch,
    #             loss_function,
    #             logger,
    #             encoder,
    #             decoder,
    #             encoder_optimizer,
    #             decoder_optimizer,
    #             prev_loss,
    #         )
    #         val_end_time = time.time()
    #         print(
    #             f"Validation loss: {model_loss}, Average distance: {avg_dis}, Total time: {(val_end_time-global_start_time)/60.0} mins"
    #         )

if __name__ == "__main__":
    main()