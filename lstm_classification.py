import os
import time
import math
import argparse
import torch
import joblib
import pandas as pd
import numpy as np
import utils.baseline_config as config
import torch.nn.functional as F
import torch
from torch import nn
from torch.optim import Adam
from typing import Tuple, Any, Dict
from torch.utils.data import Dataset, DataLoader
from logger import Logger
from shapely.geometry import Point, Polygon, LineString, LinearRing
from shapely.affinity import affine_transform, rotate
from argoverse.evaluation.eval_forecasting import compute_forecasting_metrics
from argoverse.utils.forecasting_evaluation import evaluate_prediction

from utils.baseline_config import (
    BASELINE_INPUT_FEATURES,
    BASELINE_OUTPUT_FEATURES,
    FEATURE_FORMAT,
    CLASS_FEATURE_FORMAT
)

#Input Arguments
parser = argparse.ArgumentParser(description='LSTM_core')
parser.add_argument(
    "--model_path",
    required=False,
    default= 'saved_models',
    type = str,
    help = "path to the saved model"
)

#GPU Check
cuda = torch.cuda.is_available()
if cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

#Global parameters
best_loss = float("inf")

#Define utility function
def normalize_trajectory(dataframe):
    observe_len = 20
    translation = []
    rotation = []

    normalized_traj = []
    features_data = np.stack(dataframe["FEATURES"].values)
    x = features_data[:,:,FEATURE_FORMAT["X"]].astype('float64') #shape [5,50]
    y = features_data[:,:,FEATURE_FORMAT["Y"]].astype('float64') #shape [5,50]

    num_samples  = x.shape[0]

#Load the Data
def load_and_preprocess(
    feature_file: str = "features/forecasting_features_val.pkl",
    mode: str = "train",
): 

    #load the data and save in dataframe
    dataframe = pd.read_pickle(feature_file)
    features_data = np.stack(dataframe["FEATURES"].values) #shape: [5,50,8]
    decision_data = np.stack(dataframe["GT"].values) #shape: [5,]

    #specify the desired inputs and outputs
    #TODO: change feature loaded
    input_features_list = ["DELTA_X","DELTA_Y","RELATIVE_ROT_ANGLE"]
    input_features_idx = [CLASS_FEATURE_FORMAT[feature] for feature in input_features_list]
    
    #load the features from dataframe
    input_features_data = features_data[:,:,input_features_idx].astype('float64') #shape: [5,50,8]


    _input = input_features_data[:,:20] #shape: [5,20,5]
    
    if mode == "train":
        _output = decision_data.astype(int) #shape: [5,30,2]
    else:
        _output = None

    data_dict = {
        "input": _input,
        "output": _output,
    }
    
    return data_dict

#TODO Revise the Encoder
class ClassRNN(nn.Module):
    def __init__(self, 
                 input_size:int = 5,
                 embedded_size:int = 8,
                 hidden_size:int =  32,
                 output_size:int = 1,
                 dropout:float = 0.5,
                 n_layers:int = 3):
        super().__init__()
        self.input_size = input_size
        self.embedded_size = embedded_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.linear1 = nn.Linear(self.input_size, self.embedded_size)
        self.rnn = nn.LSTM(self.embedded_size,
                           self.hidden_size,
                           num_layers = self.n_layers,
                           dropout=self.dropout)
        self.fc = nn.Linear(self.hidden_size,self.output_size)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x: torch.FloatTensor, hidden: Any):
        embedded = F.relu(self.linear1(x), inplace=False)
        output, (hn, cn) = self.rnn(embedded, hidden)
        return self.fc(hn)


def train(train_loader, epoch ,loss_function, logger,
          rnn, rnn_optimizer
):  
    for batch_idx, (_input, target) in enumerate(train_loader):
        #print(batch_idx)
        _input = _input.to(device) #[5, 20, 5]
        target = target.to(device) #[5, ]
        # print(_input[0])
        
        #set rnn to train mode
        rnn.train()

        #clear up the gradient for optimizer
        rnn.zero_grad()

        #encoder arguments
        #Input Sample shape: [5,20,5]
        #output sample shape: [5,]
        sample_size = _input.shape[0]
        input_size = _input.shape[1]
        output_size = 1
        #print(sample_size, input_size, output_size)

        #initialize the hidden state
        # >>> rnn = nn.LSTMCell(10, 20)
        # >>> input = torch.randn(6, 3, 10)
        # >>> hx = torch.randn(3, 20)
        # >>> cx = torch.randn(3, 20)
        # >>> output = []
        # >>> for i in range(6):
        #         hx, cx = rnn(input[i], (hx, cx))
        #         output.append(hx)

        hx = torch.zeros(sample_size, rnn.hidden_size).to(device)
        cx = torch.zeros(sample_size, rnn.hidden_size).to(device)
        rnn_hidden = (hx,cx)

        #Encoder observed trajectory
        predictions, (_,_) = rnn(_input, rnn_hidden)
        
        #Normalize the loss
        loss = loss_function(predictions, target)
        #Backpropagation
        loss.backward()
        rnn_optimizer.step()

        #print(f"Train -- Epoch:{epoch}, loss:{loss}")
        return loss

# #TODO Define the validation network
# def validate(val_loader, epoch, loss_function, logger,
#             encoder, decoder, encoder_optimizer, decoder_optimizer,
#             prev_loss
# ):
#     global best_loss
#     loss_list = []
#     dis_list = []

#     for i, (_input, target) in enumerate(val_loader):

#         _input = _input.to(device)
#         target = target.to(device)

#         #set encoder and decoder to validation mode
#         encoder.eval()
#         decoder.eval() 

#         #Define encoder arguments
#         sample_size = _input.shape[0]
#         input_size = _input.shape[1]
#         output_size = target.shape[1]

#         #Initialize encoder hidden state
#         hx = torch.zeros(sample_size, encoder.hidden_size).to(device)
#         cx = torch.zeros(sample_size, encoder.hidden_size).to(device)
#         encoder_hidden = (hx,cx)

#         #Encoder observed trajectory
#         for encoder_idx in range(input_size):
#             encoder_input = _input[:, encoder_idx, :]
#             encoder_hidden = encoder(encoder_input, encoder_hidden)

#         #Initialize decoder input
#         decoder_input = encoder_input[:,:2]

#         #Initialize decoder hidden state as encoder hidden state
#         decoder_hidden = encoder_hidden

#         decoder_outputs = torch.zeros((sample_size, output_size, 2)).to(device)

#         #Initialize loss
#         loss = 0

#         # Decode hidden state in future trajectory
#         for decoder_idx in range(output_size):
#             decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
#             decoder_outputs[:, decoder_idx,:] = decoder_output

#             #Accumulate the loss
#             loss += loss_function(decoder_output[:, :2], target[:, decoder_idx, :2])

#             #use own prediction as the input for next step
#             decoder_input = decoder_output
        
#         #Normalize the loss (in one batch)
#         loss = loss/output_size
#         loss_list.append(loss)

#         dis = np.sqrt(np.linalg.norm((decoder_outputs[:,-1,:]-target[:,-1,:]).detach().numpy())**2/sample_size)
#         dis_list.append(dis)

#     #Average the loss (for all batches)
#     val_loss = sum(loss_list)/len(loss_list)
#     avg_dis = sum(dis_list)/len(dis_list)

#     if val_loss <= best_loss:
#         best_loss = val_loss
        
#         save_dir = 'models'
#         os.makedirs(save_dir,exist_ok=True)
#         filename = "{}/LSTM.pth.tar".format(save_dir)

#         state = {
#                 "epoch": epoch + 1,
#                 "encoder_state_dict": encoder.state_dict(),
#                 "decoder_state_dict": decoder.state_dict(),
#                 "best_loss": val_loss,
#                 "encoder_optimizer": encoder_optimizer.state_dict(),
#                 "decoder_optimizer": decoder_optimizer.state_dict(),
#             }
#         torch.save(state, filename)
    
#     return val_loss, avg_dis



#TODO Define the inference function to calculate the prediction

# Pytorch utiliites

#TODO Correct the dataset loader
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
        if mode != "test":
            self.output_data = data_dict["output"]
        self.num_samples = self.input_data.shape[0]

        #Process parameter
        self.normalize = 0
        self.use_map = 1
        self.use_delta = 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        # datatuple = self.getitem_helpers()
        return (
            torch.FloatTensor(self.input_data[idx]),
            torch.empty(1) if self.mode == "test" else torch.FloatTensor(
                self.output_data[idx]),
        )
   

def main():
    #Directories
    train_dir = 'features/forecasting_features_val.pkl'
    val_dir = 'features/forecasting_features_val.pkl'
    test_dir = "data/test_obs/data"
    
    #Hyperparameters
    batch_size = 64
    lr = 0.001
    num_epochs = 20000
    epoch = 0

    args = parser.parse_args()
    print(f"Using all ({joblib.cpu_count()}) CPUs....")
    if cuda:
        print(f"Using all ({torch.cuda.device_count()}) GPUs...")

    #Get the data, in dictionary format
    data_dict = load_and_preprocess(train_dir)
    val_dict = load_and_preprocess(val_dir)
    # data_input = data_dict["input"]
    # data_output = data_dict["output"]
    # print(data_input.shape)

    #Get the model
    loss_function = nn.MSELoss()
    encoder = LSTMEncoder(input_size=5)
    decoder = LSTMDecoder(output_size=2)
    encoder.to(device)
    decoder.to(device)

    #TODO change to correct optimizers
    encoder_optimizer = Adam(encoder.parameters(), lr = lr)
    decoder_optimizer = Adam(decoder.parameters(), lr = lr)

    #Training
    #Logger
    log_dir = os.path.join(os.getcwd(), "lstm_logs")
    #Transform data to Pytorch Dataset format
    train_dataset = Dataset_Loader(data_dict, "train")
    val_dataset = Dataset_Loader(val_dict, "val")
    print(len(train_dataset))
    
    # print(train_dataset[0][0].shape) #shape: [20, 5]
    # print(train_dataset[0][1].shape) #shape: [30, 2]
    # print(train_dataset[0][0])

    #Setting Dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        drop_last = False,
        shuffle = False,
    )

    #print(train_loader.shape)

    val_loader = DataLoader(
        val_dataset,
        batch_size = batch_size,
        drop_last = False,
        shuffle = False,
    )

    print("Training begins ...")
    
    #logger = Logger(log_dir, name="logfile")
    logger = log_dir
    global_start_time = time.time()
    best_loss = float("inf")
    prev_loss = best_loss
    while epoch < num_epochs:
        start = time.time()
        loss = train(
            train_loader,
            epoch,
            loss_function,
            logger,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
        )
        end = time.time()

        if epoch % 100==0:
            print(
                f"Epoch: {epoch} completed in: {end-start}s, Total time: {(end - global_start_time) /60.0} mins, loss is: {loss}"
            )

        epoch+=1
        if epoch % 500==0:
            val_start_time = time.time()
            model_loss, avg_dis = validate(
                val_loader,
                epoch,
                loss_function,
                logger,
                encoder,
                decoder,
                encoder_optimizer,
                decoder_optimizer,
                prev_loss,
            )
            val_end_time = time.time()
            print(
                f"Validation loss: {model_loss}, Average distance: {avg_dis}, Total time: {(val_end_time-global_start_time)/60.0} mins"
            )

if __name__ == "__main__":
    main()