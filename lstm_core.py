import os
import time
import argparse
import torch
import joblib
import pandas as pd
import numpy as np
import utils.baseline_utils as baseline_utils
import utils.baseline_config as config
import torch.nn.functional as F
import torch
from torch import nn
from torch.optim import Adam
from typing import Tuple, Any, Dict
from joblib import Parallel, delayed
from torch.utils.data import Dataset, DataLoader
from utils.lstm_utils import ModelUtils, LSTMDataset
from logger import Logger


from utils.baseline_config import (
    BASELINE_INPUT_FEATURES,
    BASELINE_OUTPUT_FEATURES,
    FEATURE_FORMAT,
    LSTM_DICT_NAME,
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

#Load the Data
def load_and_preprocess(
    feature_file: str = "features/forecasting_features_val.pkl",
    mode: str = "train",
): 

    #load the data and save in dataframe
    dataframe = pd.read_pickle(feature_file)
    features_data = np.stack(dataframe["FEATURES"].values) #shape: [5,50,11]

    #TODO: Normalize the data
    #TODO: Select input and output features

    #specify the desired inputs and outputs
    input_features_list = ["MIN_DISTANCE_FRONT","MIN_DISTANCE_BACK", "NUM_NEIGHBORS","OFFSET_FROM_CENTERLINE","DISTANCE_ALONG_CENTERLINE"]
    input_features_idx = [FEATURE_FORMAT[feature] for feature in input_features_list]
    
    #load the features from dataframe
    input_features_data = features_data[:,:,input_features_idx].astype('float64') #shape: [5,50,5]


    _input = input_features_data[:,:20] #shape: [5,20,5]
    
    if mode == "train":
        output_feastures_list = ["OFFSET_FROM_CENTERLINE","DISTANCE_ALONG_CENTERLINE"]
        ouput_features_idx = [FEATURE_FORMAT[feature] for feature in output_feastures_list]
        output_feastures_data = features_data[:,:,ouput_features_idx].astype('float64')
        _output = output_feastures_data[:,20:] #shape: [5,20,2]
    else:
        _output = None

    data_dict = {
        "input": _input,
        "output": _output,
    }
    
    return data_dict

#TODO Revise the Encoder
class LSTMEncoder(nn.Module):
    def __init__(self, 
                 input_size:int = 2,
                 embedding_size:int = 8,
                 hidden_size:int =  16):
        super().__init__()
        self.hidden_size = hidden_size
        #torch.nn.Linear(in_features, out_features, bias=True)
        self.linear1 = nn.Linear(input_size, embedding_size,bias=True)
        #torch.nn.LSTMCell(input_size, hidden_size, bias=True)
        self.lstm1 = nn.LSTMCell(embedding_size,hidden_size,bias=True)

    def forward(self, x: torch.FloatTensor, hidden: Any):
        embedded = F.relu(self.linear1(x), inplace=False)
        hidden = self.lstm1(embedded, hidden)
        return hidden

#TODO Revise the Decoder
class LSTMDecoder(nn.Module):
    def __init__(self,
                 embedding_size=8,
                 hidden_size=16,
                 output_size=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(output_size,embedding_size)
        self.lstm1 = nn.LSTMCell(embedding_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size,output_size)
    
    def forward(self, x, hidden):
        embedded = F.relu(self.linear1(x))
        hidden = self.lstm1(embedded, hidden)
        output = self.linear2(hidden[0])
        return output, hidden

#TODO Define training network
def train(train_loader, epoch ,loss_function, logger,
          encoder, decoder, encoder_optimizer, decoder_optimizer,
):  
    for i, (_input, target) in enumerate(train_loader):
        _input = _input.to(device)
        target = target.to(device)
        
        #set encoder and decoder to train mode
        encoder.train()
        decoder.train()

        #clear up the gradient for optimizer
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        #encoder arguments
        #Input Sample shape: [5,50,11]
        #output sample shape: [5,50,2]
        sample_size = _input.shape[0]
        input_size = _input.shape[1]
        output_size = target.shape[1]

        #initialize the hidden state
        encoder_hidden = (torch.zeros(sample_size,encoder.hidden_size).to(device),
                          torch.zeros(sample_size,encoder.hidden_size).to(device))

#TODO Define the validation network
def validate(train_loader, epoch, loss_function, logger,
            encoder, decoder, encoder_optimizer, decoder_optimizer,
            prev_loss
):
    pass

#TODO Define the inference function to calculate the prediction

# Pytorch utiliites
def my_collate_fn(self, batch):
        # _input, output, helpers = [], [], []
        _input, output= [], []

        for item in batch:
            _input.append(item[0])
            output.append(item[1])
            # helpers.append(item[2])
        _input = torch.stack(_input)
        output = torch.stack(output)
        # return [_input, output, helpers]
        return [_input, output]

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
        self.normalize = 1
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
            # self.datatuple[idx],
        )

    # def getitem_helpers(self) -> Tuple[Any]:
    #     """Get helpers for running baselines.

    #     Returns:
    #         helpers: Tuple in the format specified by LSTM_HELPER_DICT_IDX

    #     Note: We need a tuple because DataLoader needs to index across all these helpers simultaneously.

    #     """
    #     helper_df = self.data_dict["dataframe"]
    #     candidate_centerlines = helper_df["CANDIDATE_CENTERLINES"].values
    #     candidate_nt_distances = helper_df["CANDIDATE_NT_DISTANCES"].values
    #     xcoord = np.stack(helper_df["FEATURES"].values
    #                       )[:, :, config.FEATURE_FORMAT["X"]].astype("float")
    #     ycoord = np.stack(helper_df["FEATURES"].values
    #                       )[:, :, config.FEATURE_FORMAT["Y"]].astype("float")
    #     centroids = np.stack((xcoord, ycoord), axis=2)
    #     _DEFAULT_HELPER_VALUE = np.full((centroids.shape[0]), None)
    #     city_names = np.stack(helper_df["FEATURES"].values
    #                           )[:, :, config.FEATURE_FORMAT["CITY_NAME"]]
    #     seq_paths = helper_df["SEQUENCE"].values
    #     translation = (helper_df["TRANSLATION"].values
    #                    if self.normalize else _DEFAULT_HELPER_VALUE)
    #     rotation = (helper_df["ROTATION"].values
    #                 if self.normalize else _DEFAULT_HELPER_VALUE)

    #     use_candidates = self.use_map and self.mode == "test"

    #     candidate_delta_references = (
    #         helper_df["CANDIDATE_DELTA_REFERENCES"].values
    #         if self.use_map and use_candidates else _DEFAULT_HELPER_VALUE)
    #     delta_reference = (helper_df["DELTA_REFERENCE"].values
    #                        if self.use_delta and not use_candidates else
    #                        _DEFAULT_HELPER_VALUE)

    #     helpers = [None for i in range(len(config.LSTM_HELPER_DICT_IDX))]

    #     # Name of the variables should be the same as keys in LSTM_HELPER_DICT_IDX
    #     for k, v in config.LSTM_HELPER_DICT_IDX.items():
    #         helpers[v] = locals()[k.lower()]

    #     return tuple(helpers)
   

def main():
    #Directories
    train_dir = "data/train/data"
    val_dir = "data/val/data"
    test_dir = "data/test_obs/data"
    
    #Hyperparameters
    batch_size = 256
    lr = 0.001
    num_epochs = 10
    epoch = 0

    args = parser.parse_args()
    print(f"Using all ({joblib.cpu_count()}) CPUs....")
    if cuda:
        print(f"Using all ({torch.cuda.device_count()}) GPUs...")
    #Define the model
    model_utils = ModelUtils()

    #Get the data, in dictionary format
    data_dict = load_and_preprocess()
    val_dict = load_and_preprocess()
    # data_input = data_dict["input"]
    # data_output = data_dict["output"]
    # print(data_input.shape)

    #Get the model
    loss_function = nn.MSELoss()
    encoder = LSTMEncoder(2, 8, 16)
    decoder = LSTMDecoder(8,16,2)
    encoder.to(device)
    decoder.to(device)

    #TODO change to correct optimizers
    encoder_optimizer = Adam(encoder.parameters(), lr = lr)
    decoder_optimizer = Adam(decoder.parameters(), lr = lr)
    # encoder_optimizer = 0
    # decoder_optimizer = 0
    #Training
    #Logger
    log_dir = os.path.join(os.getcwd(), "lstm_logs")
    #Transform data to Pytorch Dataset format
    train_dataset = Dataset_Loader(data_dict, "train")
    val_dataset = Dataset_Loader(val_dict, "val")
    print(len(train_dataset))
    
    #TODO Test dataloader
    print(train_dataset[0][0].shape)

    #Setting Dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        drop_last = False,
        shuffle = False,
    )

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
        train(
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

        print(
            f"Epoch: {epoch} completed in: {end-start}s, Total time: {(end - global_start_time) /60.0} mins"
        )

        epoch+=1
        if epoch % 5==0:
            val_start_time = time.time()
            model_loss = validate(
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
                f"Validation loss: {prev_loss}, Total time: {(val_end_time-global_start_time)/60.0} mins"
            )

if __name__ == "__main__":
    main()