import os
import time
import math
import argparse
import torch
import joblib
import pandas as pd
import numpy as np
import baseline_config as config
import torch.nn.functional as F
import torch
from torch import nn
from torch.optim import Adam
from typing import Tuple, Any, Dict
from torch.utils.data import Dataset, DataLoader
# from shapely.geometry import Point, Polygon, LineString, LinearRing
# from shapely.affinity import affine_transform, rotate

from baseline_config import (
    BASELINE_INPUT_FEATURES,
    BASELINE_OUTPUT_FEATURES,
    FEATURE_FORMAT,
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
# def normalize_trajectory(dataframe):
#     observe_len = 20
#     translation = []
#     rotation = []

#     normalized_traj=[]
#     features_data = dataframe.values
#     x = features_data[:,:,FEATURE_FORMAT["X"]].astype('float64') #shape [5,50]
#     y = features_data[:,:,FEATURE_FORMAT["Y"]].astype('float64') #shape [5,50]

#     num_samples  = x.shape[0]

#     #Normalize the trajectory for each sample
#     for i in range(num_samples):
#         #In each sample
#         xy_tuple = np.stack((x[i],y[i]), axis = 1) #shape [50,2]
#         traj = LineString(xy_tuple)
#         start = traj[0] #shape [2,]

#         #Translation normalization: start with (0.0, 0.0)
#         #[a, b, c, d, x, y] is the planner transformation matrix[a, b, x; c, d, y; 0, 0, 1]
#         g = [1, 0, 0, 1, -start[0], -start[1]]
#         traj_trans = affine_transform(traj, g)
        
#         #Rotation normalization: 
#         end = traj_trans.coords[observe_len-1]
#         if end[0]==0 and end[1]==0:
#             angle = 0.0
#         elif end[0]==0:
#             angle = 90.0 if end[1]<0 else -90.0
#         elif end[1]==0:
#             angle = 0.0 if end[0]>0 else 180.0
#         else:
#             angle = math.degrees(math.atan(end[1]/end[0]))
#             if (end[0] > 0 and end[1] > 0) or (end[0] > 0 and end[1] < 0):
#                 angle = -angle
#             else:
#                 angle = 180.0 - angle
#         #Rotate normalization: end with y=0
#         traj_rotate = rotate(traj_trans, angle, origin=(0,0)).coords[:]

#         #Transform to numpy
#         traj_norm = np.array(traj_rotate)

#         #Append to containers
#         translation.append(g)
#         rotation.append(angle)
#         normalized_traj.append(traj_norm)
    
#     #update the dataframe and return normalized trajectory
#     dataframe["TRANSLATION"] = translation
#     dataframe["ROTATION"] = rotation
#     return np.stack(normalized_traj)


#Load the Data
def load_and_preprocess(
    feature_file: str = "features/forecasting_features_val.pkl",
    mode: str = "train",
): 

    #load the data and save in dataframe
    dataframe = pd.read_pickle(feature_file)
    features_data = np.stack(dataframe["FEATURES"].values) #shape: [5,50,11]

    #Normalize the trajectory (only for without map)
    #norm_traj = normalize_trajectory(dataframe)

    #specify the desired inputs and outputs
    input_features_list = ["OFFSET_FROM_CENTERLINE","DISTANCE_ALONG_CENTERLINE","NUM_NEIGHBORS",
                           "MIN_DISTANCE_FRONT","MIN_DISTANCE_BACK", 
                           "MIN_DISTANCE_FRONT_VEL","MIN_DISTANCE_BACK_VEL",
                           "NEIGHBORS_MEAN_VEL","NEIGHBORS_MAX_VEL","NEIGHBORS_MIN_VEL",
                           "RELATIVE_ROT_ANGLE","ANGLE_W_CL"]
    input_features_idx = [FEATURE_FORMAT[feature] for feature in input_features_list]
    # input_features_idx = [9,10,6,7,8]
    #load the features from dataframe
    input_features_data = features_data[:,:,input_features_idx].astype('float64') #shape: [5,50,5]

    _input = input_features_data[:,:20] #shape: [5,20,5]
    
    if mode == "train":
        output_feastures_list = ["OFFSET_FROM_CENTERLINE","DISTANCE_ALONG_CENTERLINE"]
        ouput_features_idx = [FEATURE_FORMAT[feature] for feature in output_feastures_list]
        # ouput_features_idx = [9,10]
        output_feastures_data = features_data[:,:,ouput_features_idx].astype('float64')
        _output = output_feastures_data[:,20:] #shape: [5,30,2]
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
                 input_size:int = 12,
                 embedding_size:int = 16,
                 hidden_size:int =  32):
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
                 embedding_size=16,
                 hidden_size=32,
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
    loss_list = []
    for batch_idx, (_input, target) in enumerate(train_loader):
        #print(batch_idx)
        _input = _input.to(device) #[5, 20, 5]
        target = target.to(device) #[5, 30, 2]
        # print(_input[0])
        
        #set encoder and decoder to train mode
        encoder.train()
        decoder.train()

        #clear up the gradient for optimizer
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        #encoder arguments
        #Input Sample shape: [5,20,5]
        #output sample shape: [5,30,2]
        sample_size = _input.shape[0]
        input_size = _input.shape[1]
        output_size = target.shape[1]
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

        hx = torch.zeros(sample_size, encoder.hidden_size).to(device)
        cx = torch.zeros(sample_size, encoder.hidden_size).to(device)
        encoder_hidden = (hx,cx)

        #Encoder observed trajectory
        for encoder_idx in range(input_size):
            encoder_input = _input[:, encoder_idx, :]
            encoder_hidden = encoder(encoder_input, encoder_hidden)

        #Initialize decoder input
        decoder_input = encoder_input[:,:2]

        #Initialize decoder hidden state as encoder hidden state
        decoder_hidden = encoder_hidden

        decoder_outputs = torch.zeros((sample_size, output_size, 2)).to(device)

        #Initialize the losses
        loss = 0

        # Decode hidden state in future trajectory
        for decoder_idx in range(output_size):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            decoder_outputs[:, decoder_idx,:] = decoder_output

            #Accumulate the loss
            loss += loss_function(decoder_output[:, :2], target[:, decoder_idx, :2])

            #use own prediction as the input for next step
            decoder_input = decoder_output
        
        #Normalize the loss
        loss = loss/output_size
        loss_list.append(loss)
        #Backpropagation
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

    train_loss = sum(loss_list)/len(loss_list)
    return train_loss

#TODO Define the validation network
def validate(val_loader, epoch, loss_function, logger,
            encoder, decoder, encoder_optimizer, decoder_optimizer,
            prev_loss
):
    global best_loss
    loss_list = []
    dis_list = []
    ADE_list = []

    for i, (_input, target) in enumerate(val_loader):

        _input = _input.to(device)
        target = target.to(device)

        #set encoder and decoder to validation mode
        encoder.eval()
        decoder.eval() 

        #Define encoder arguments
        sample_size = _input.shape[0]
        input_size = _input.shape[1]
        output_size = target.shape[1]

        #Initialize encoder hidden state
        hx = torch.zeros(sample_size, encoder.hidden_size).to(device)
        cx = torch.zeros(sample_size, encoder.hidden_size).to(device)
        encoder_hidden = (hx,cx)

        #Encoder observed trajectory
        for encoder_idx in range(input_size):
            encoder_input = _input[:, encoder_idx, :]
            encoder_hidden = encoder(encoder_input, encoder_hidden)

        #Initialize decoder input
        decoder_input = encoder_input[:,:2]

        #Initialize decoder hidden state as encoder hidden state
        decoder_hidden = encoder_hidden

        decoder_outputs = torch.zeros((sample_size, output_size, 2)).to(device)

        #Initialize loss
        loss = 0

        # Decode hidden state in future trajectory
        for decoder_idx in range(output_size):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            decoder_outputs[:, decoder_idx,:] = decoder_output

            #Accumulate the loss
            loss += loss_function(decoder_output[:, :2], target[:, decoder_idx, :2])

            #use own prediction as the input for next step
            decoder_input = decoder_output
        
        #Normalize the loss (in one batch)
        loss = loss/output_size
        loss_list.append(loss)

        dis = np.sqrt(np.linalg.norm((decoder_outputs[:,-1,:]-target[:,-1,:]).detach().cpu().clone().numpy())**2/sample_size)
        ade = np.sqrt(np.linalg.norm((decoder_outputs-target).detach().cpu().clone().numpy())**2/output_size/sample_size)
        dis_list.append(dis)
        ADE_list.append(ade)

    #Average the loss (for all batches)
    val_loss = sum(loss_list)/len(loss_list)
    FDE = sum(dis_list)/len(dis_list)
    ADE = sum(ADE_list)/len(ADE_list)

    if val_loss <= best_loss:
        best_loss = val_loss
        
        save_dir = 'models'
        os.makedirs(save_dir,exist_ok=True)
        filename = "{}/LSTM.pth.tar".format(save_dir)

        state = {
                "epoch": epoch + 1,
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "best_loss": val_loss,
                "encoder_optimizer": encoder_optimizer.state_dict(),
                "decoder_optimizer": decoder_optimizer.state_dict(),
            }
        torch.save(state, filename)
    
    return val_loss, FDE, ADE



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
            # self.datatuple[idx],
        )

    def getitem_helpers(self) -> Tuple[Any]:

        helper_df = self.data_dict["dataframe"]
        candidate_centerlines = helper_df["CANDIDATE_CENTERLINES"].values
        candidate_nt_distances = helper_df["CANDIDATE_NT_DISTANCES"].values
        xcoord = np.stack(helper_df["FEATURES"].values
                          )[:, :, config.FEATURE_FORMAT["X"]].astype("float")
        ycoord = np.stack(helper_df["FEATURES"].values
                          )[:, :, config.FEATURE_FORMAT["Y"]].astype("float")
        centroids = np.stack((xcoord, ycoord), axis=2)
        _DEFAULT_HELPER_VALUE = np.full((centroids.shape[0]), None)
        city_names = np.stack(helper_df["FEATURES"].values
                              )[:, :, config.FEATURE_FORMAT["CITY_NAME"]]
    #     seq_paths = helper_df["SEQUENCE"].values
        translation = (helper_df["TRANSLATION"].values
                       if self.normalize else _DEFAULT_HELPER_VALUE)
        rotation = (helper_df["ROTATION"].values
                    if self.normalize else _DEFAULT_HELPER_VALUE)

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
    train_dir = '../all_features/features_train.pkl'
    val_dir = '../all_features/features_val.pkl'
    # test_dir = "data/test_obs/data"
    
    #Hyperparameters
    batch_size = 64
    lr = 0.001
    num_epochs = 30
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
    encoder = LSTMEncoder(input_size=12)
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

        if epoch % 1==0:
            print(
                f"Epoch: {epoch} completed in: {end-start}s, Total time: {(end - global_start_time) /60.0} mins, loss is: {loss}"
            )

        epoch+=1
        if epoch % 3==0:
            val_start_time = time.time()
            model_loss, FDE, ADE = validate(
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
                f"Validation loss: {model_loss}, FDE: {FDE}, ADE: {ADE}, Total time: {(val_end_time-global_start_time)/60.0} mins"
            )

if __name__ == "__main__":
    main()
