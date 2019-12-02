import numpy as np
import pandas as pd
import pickle as pkl
import sys
import copy

from typing import Dict
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.centerline_utils import (
    get_nt_distance,
    get_oracle_from_candidate_centerlines
)

import os
import time
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from baseline_config import (
        RAW_DATA_FORMAT,
        _MAX_SEARCH_RADIUS_CENTERLINES,
        TEST_FEATURE_FORMAT)
from argoverse.utils.mpl_plotting_utils import visualize_centerline
from our_social_features import SocialFeaturesUtils
from our_class_features import compute_class_features
from compute_classifier_feature import compute_best_candidates
from compute_test_features import (
        find_oracle_centerline,
        compute_map_features,
        )
from lstm_evaluation import(
        get_nt_abs_values,
        get_xy_from_nt_seq,
        )
import torch
from lstm_classification import ClassRNN
from lstm_core import(
        LSTMEncoder,
        LSTMDecoder,
        )

obs_len = 20
pred_len = 30

classifier_model_dir = "models/Classifier.pth.tar"
lstm_model_dir = "models/LSTM_2.pth.tar"
data_dir = "../val/data"

def lstm_eval(data,encoder, decoder):
    
    features_data = data[1]
    cl_list = [data[2]]
    delta_ref = data[4].reshape((1,1,2))
    input_features_list = ["OFFSET_FROM_CENTERLINE","DISTANCE_ALONG_CENTERLINE","NUM_NEIGHBORS",
                           "MIN_DISTANCE_FRONT","MIN_DISTANCE_BACK", 
                           "MIN_DISTANCE_FRONT_VEL","MIN_DISTANCE_BACK_VEL",
                           "NEIGHBORS_MEAN_VEL","NEIGHBORS_MAX_VEL","NEIGHBORS_MIN_VEL",
                           "RELATIVE_ROT_ANGLE","ANGLE_W_CL"]
    input_features_idx = [TEST_FEATURE_FORMAT[feature] for feature in input_features_list]
#    input_features_idx = [9,10,6,7,8]
    #load the features from dataframe
    input_features_data = features_data[:,input_features_idx].astype('float64') #shape: [5,50,5]
    input_data = input_features_data[:20,:] #shape: [5,20,5]
    
    ground_truth_list = ["X","Y"]
    ground_truth_idx = [TEST_FEATURE_FORMAT[feature] for feature in ground_truth_list]

    ground_truth_data = features_data[:,ground_truth_idx].astype('float64')
    ground_truth = ground_truth_data[20:,:].reshape((1,30,2)) #shape: [5,30,2]
    
    input_data = torch.from_numpy(input_data.reshape((1,20,12))).float()

    #Define encoder arguments
    sample_size = input_data.shape[0]
    input_size = input_data.shape[1]
    output_size = 30

    #Initialize encoder hidden state
    hx = torch.zeros(sample_size, encoder.hidden_size)
    cx = torch.zeros(sample_size, encoder.hidden_size)
    encoder_hidden = (hx,cx)

    #Encoder observed trajectory
    for encoder_idx in range(input_size):
        encoder_input = input_data[:, encoder_idx, :]
        encoder_hidden = encoder(encoder_input, encoder_hidden)

    #Initialize decoder input
    decoder_input = encoder_input[:,:2]

    #Initialize decoder hidden state as encoder hidden state
    decoder_hidden = encoder_hidden

    decoder_outputs = torch.zeros((sample_size, output_size, 2))


    # Decode hidden state in future trajectory
    for decoder_idx in range(output_size):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        decoder_outputs[:, decoder_idx,:] = decoder_output

        #use own prediction as the input for next step
        decoder_input = decoder_output
    
    nt_abs = get_nt_abs_values(
            decoder_outputs.detach().cpu().clone().numpy().reshape((1,30,2)),
            delta_ref
            )
    xy_abs = get_xy_from_nt_seq(nt_abs,cl_list)
    
    dis = np.linalg.norm((xy_abs[:,-1,:]-ground_truth[:,-1,:]),axis=1)
    ade = np.sum(np.linalg.norm((xy_abs-ground_truth),axis=2),axis=1)/output_size
    
    data.append(xy_abs.reshape((30,2)))
    data.append(dis)
    data.append(ade)
    
    return data

def viz_trajectory(data):
    
    xy_pred = data[5]
    features_data = data[1]
    ground_truth_list = ["X","Y"]
    ground_truth_idx = [TEST_FEATURE_FORMAT[feature] for feature in ground_truth_list]
    ground_truth = features_data[:,ground_truth_idx].astype('float64')
    oracle_centerline = data[2]
    
    plt.figure(figsize=(8, 7))
    visualize_centerline(oracle_centerline)
    plt.plot(
        ground_truth[:21, 0],
        ground_truth[:21, 1],
        "-",
        color="b",
        alpha=1,
        linewidth=3,
        zorder=15,
    )
    plt.plot(
        ground_truth[20:, 0],
        ground_truth[20:, 1],
        "-",
        color="k",
        alpha=1,
        linewidth=3,
        zorder=15,
    )
    plt.plot(
        xy_pred[:, 0],
        xy_pred[:, 1],
        "-",
        color="#d33e4c",
        alpha=1,
        linewidth=3,
        zorder=15,
    )

    final_x = xy_pred[-1, 0]
    final_y = xy_pred[-1, 1]

    plt.plot(
        final_x,
        final_y,
        "o",
        color="#d33e4c",
        alpha=1,
        markersize=10,
        zorder=15,
    )
    plt.xlabel("Map X")
    plt.ylabel("Map Y")
    plt.axis("off")
    plt.axis("equal")
    plt.legend(["Oracle Centerline","Observed trajectory","Ground Truth","Predicted Trajectory","End Point"])
    plt.title("Trajectory forcasting with ADE = {:.2f} and FDE = {:.2f}".format(data[7][0],data[6][0]))
    plt.show()
    
def load_compute_save(file_name, social_instance, rnn, encoder, decoder, data_dir):
    data = []
    file_path = data_dir+'/'+file_name
    df = pd.read_csv(file_path, dtype={"TIMESTAMP": str})
    agent_track = df[df["OBJECT_TYPE"] == "AGENT"].values

    class_features = compute_class_features(df,agent_track,obs_len,obs_len+pred_len,RAW_DATA_FORMAT)
    
    best_str, best_turn = compute_best_candidates(
        agent_track,
        obs_len,
        pred_len,
        RAW_DATA_FORMAT
    )
    
    oracle_centerline_label = find_oracle_centerline(
            class_features,
            rnn
            )
    
    if oracle_centerline_label == 0:
        oracle_centerline = best_turn
        second_centerline = best_str
    else:
        oracle_centerline = best_str
        second_centerline = best_turn
    
    map_features, delta_ref = compute_map_features(
        agent_track,
        oracle_centerline,
        obs_len,
        pred_len,
        RAW_DATA_FORMAT,
    )
    
    social_features = social_instance.compute_social_features(
        df, agent_track, obs_len, obs_len + pred_len,
        RAW_DATA_FORMAT)
    #print(agent_track.shape,social_features.shape,map_features.shape,class_features.shape)
    if agent_track.shape[0] == obs_len:
        features = np.concatenate((agent_track,social_features,map_features,class_features[:,(0,1,3,4,5,6,7)]),axis=1)
    else:
        features = np.concatenate((map_features,class_features[:,(0,1,3,4,5,6,7)]),axis=1)
        features = np.concatenate((features,np.full((pred_len,features.shape[1]),None)),axis=0)
        features = np.concatenate((agent_track,social_features,features),axis=1)
    #print(agent_track.shape,social_features.shape,map_features.shape,class_features.shape)
    
    name_id = int(file_name.split(".")[0])

    data = [name_id, features, oracle_centerline, second_centerline, delta_ref]
    
    data = lstm_eval(data,encoder,decoder)
    
    viz_trajectory(data)
    
    return data
    
if __name__ == "__main__":
    
    rnn_model = torch.load(classifier_model_dir,map_location=torch.device('cpu'))
    rnn = ClassRNN()
    rnn.load_state_dict(rnn_model["rnn_state_dict"])
    rnn.eval()
    
    model = torch.load(lstm_model_dir,map_location=torch.device('cpu'))
    encoder = LSTMEncoder()
    decoder = LSTMDecoder()
    encoder.load_state_dict(model['encoder_state_dict'])
    encoder.eval()
    decoder.load_state_dict(model['decoder_state_dict'])
    decoder.eval()
    
    social_instance = SocialFeaturesUtils()
    
    file_names = os.listdir(data_dir)
    ind = np.random.randint(len(file_names))
    file_name = file_names[ind]
    data = load_compute_save(file_name, social_instance, rnn, encoder, decoder, data_dir)