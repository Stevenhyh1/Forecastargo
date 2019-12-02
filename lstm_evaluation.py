#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 11:03:00 2019

@author: x
"""

import math
import pandas as pd
import numpy as np
import time

from typing import Any, Dict, List, Optional, Tuple, Union
from shapely.geometry import Point, Polygon, LineString, LinearRing
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from argoverse.utils.mpl_plotting_utils import visualize_centerline
from argoverse.evaluation.competition_util import generate_forecasting_h5
from baseline_config import (
        FEATURE_FORMAT,
        TEST_FEATURE_FORMAT,
        )
from lstm_core import(
        LSTMEncoder,
        LSTMDecoder,
        )

def my_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    _id = np.stack([item[0] for item in batch])
    _input = torch.from_numpy(np.stack([item[1] for item in batch])).float()
    _output = torch.from_numpy(np.stack([item[2] for item in batch])).float()
    _gt = np.stack([item[3] for item in batch])
    _cl = [item[4] for item in batch]
    _delta = np.stack([item[5] for item in batch])
    return (_id,_input,_output,_gt,_cl,_delta)
        
def load_and_preprocess(
    feature_file: str,
    feature_format
): 

    #load the data and save in dataframe
    dataframe = pd.read_pickle(feature_file)
    features_data = np.stack(dataframe["FEATURES"].values) #shape: [5,50,11]
    centerlines = dataframe['ORACLE_CENTERLINES'].values
    delta_ref = np.stack(dataframe["DELTA_REFERENCE"].values) #shape: [5,2]
    seq_id = dataframe['ID'].values
    
    #specify the desired inputs and outputs
    input_features_list = ["OFFSET_FROM_CENTERLINE","DISTANCE_ALONG_CENTERLINE","NUM_NEIGHBORS",
                           "MIN_DISTANCE_FRONT","MIN_DISTANCE_BACK", 
                           "MIN_DISTANCE_FRONT_VEL","MIN_DISTANCE_BACK_VEL",
                           "NEIGHBORS_MEAN_VEL","NEIGHBORS_MAX_VEL","NEIGHBORS_MIN_VEL",
                           "RELATIVE_ROT_ANGLE","ANGLE_W_CL"]
    input_features_idx = [FEATURE_FORMAT[feature] for feature in input_features_list]
#    input_features_idx = [9,10,6,7,8]
    #load the features from dataframe
    input_features_data = features_data[:,:,input_features_idx].astype('float64') #shape: [5,50,5]
    _input = input_features_data[:,:20,:] #shape: [5,20,5]
    
    output_feastures_list = ["OFFSET_FROM_CENTERLINE","DISTANCE_ALONG_CENTERLINE"]
    ouput_features_idx = [FEATURE_FORMAT[feature] for feature in output_feastures_list]
#    ouput_features_idx = [9,10]
    output_feastures_data = features_data[:,:,ouput_features_idx].astype('float64')
    _output = output_feastures_data[:,20:] #shape: [5,30,2]
    
    ground_truth_list = ["X","Y"]
    ground_truth_idx = [FEATURE_FORMAT[feature] for feature in ground_truth_list]

    ground_truth_data = features_data[:,:,ground_truth_idx].astype('float64')
    _gt = ground_truth_data[:,20:,:] #shape: [5,30,2]
    
    data_dict = {
        "id": seq_id,
        "input": _input,
        "output": _output,
        "ground_truth": _gt,
        "centerlines": centerlines,
        "delta_reference": delta_ref,
    }
#    print(_input.shape,_output.shape,_gt.shape,centerlines[0].shape,delta_ref.shape)
    return data_dict

class data_loader(Dataset):
    def __init__(self, data_dict):

        #Get the data
        self.input_data = data_dict["input"]
        self.output_data = data_dict["output"]
        self.ground_truth = data_dict['ground_truth']
        self.centerlines = data_dict["centerlines"]
        self.delta_ref = data_dict["delta_reference"]
        self.seq_id = data_dict["id"]
        
        self.num_samples = self.input_data.shape[0]

        #Process parameter
        self.normalize = 0
        self.use_map = 1
        self.use_delta = 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        return (
            self.seq_id[idx],
            self.input_data[idx],
            self.output_data[idx],
            self.ground_truth[idx],
            self.centerlines[idx],
            self.delta_ref[idx],
        )

def get_nt_abs_values(
        pred_coords: np.ndarray,
        delta_ref: np.ndarray,
        ):
#    print(obs_coords.shape,pred_coords.shape)
#    nt_abs_coords = np.concatenate((obs_coords,pred_coords),axis=1)
#    nt_abs_coords[:,0,:] = delta_ref
#    for i in range(1,nt_abs_coords.shape[1]):
#        nt_abs_coords[:,i,:] += nt_abs_coords[:,i-1,:]
#    
#    return nt_abs_coords[:,20:,:]
    for i in range(pred_coords.shape[0]):
        pred_coords[i,:,:] += delta_ref[i,:]
        
    return pred_coords

def get_xy_from_nt_seq(nt_seq: np.ndarray,
                       centerlines: List[np.ndarray]) -> np.ndarray:
    """Convert n-t coordinates to x-y, i.e., convert from centerline curvilinear coordinates to map coordinates.

    Args:
        nt_seq (numpy array): Array of shape (num_tracks x seq_len x 2) where last dimension has 'n' (offset from centerline) and 't' (distance along centerline)
        centerlines (list of numpy array): Centerline for each track
    Returns:
        xy_seq (numpy array): Array of shape (num_tracks x seq_len x 2) where last dimension contains coordinates in map frame

    """
    seq_len = nt_seq.shape[1]

    # coordinates obtained by interpolating distances on the centerline
    xy_seq = np.zeros(nt_seq.shape)
    for i in range(nt_seq.shape[0]):
        curr_cl = centerlines[i]
#        line_string = LineString(curr_cl)
        for t in range(seq_len):

            # Project nt to xy
            offset_from_cl = nt_seq[i][t][0]
            dist_along_cl = nt_seq[i][t][1]
            x_coord, y_coord = get_xy_from_nt(offset_from_cl, dist_along_cl,
                                              curr_cl)
            xy_seq[i, t, 0] = x_coord
            xy_seq[i, t, 1] = y_coord

    return xy_seq


def get_xy_from_nt(n: float, t: float,
                   centerline: np.ndarray) -> Tuple[float, float]:
    """Convert a single n-t coordinate (centerline curvilinear coordinate) to absolute x-y.

    Args:
        n (float): Offset from centerline
        t (float): Distance along the centerline
        centerline (numpy array): Centerline coordinates
    Returns:
        x1 (float): x-coordinate in map frame
        y1 (float): y-coordinate in map frame

    """
    line_string = LineString(centerline)

    # If distance along centerline is negative, keep it to the start of line
    point_on_cl = line_string.interpolate(
        t) if t > 0 else line_string.interpolate(0)
    local_ls = None

    # Find 2 consective points on centerline such that line joining those 2 points
    # contains point_on_cl
    for i in range(len(centerline) - 1):
        pt1 = centerline[i]
        pt2 = centerline[i + 1]
        ls = LineString([pt1, pt2])
        if ls.distance(point_on_cl) < 1e-8:
            local_ls = ls
            break

    assert local_ls is not None, "XY from N({}) T({}) not computed correctly".format(
        n, t)

    pt1, pt2 = local_ls.coords[:]
    x0, y0 = point_on_cl.coords[0]

    # Determine whether the coordinate lies on left or right side of the line formed by pt1 and pt2
    # Find a point on either side of the line, i.e., (x1_1, y1_1) and (x1_2, y1_2)
    # If the ring formed by (pt1, pt2, (x1_1, y1_1)) is counter clockwise, then it lies on the left

    # Deal with edge cases
    # Vertical
    if pt1[0] == pt2[0]:
        m = 0
        x1_1, x1_2 = x0 + n, x0 - n
        y1_1, y1_2 = y0, y0
    # Horizontal
    elif pt1[1] == pt2[1]:
        m = float("inf")
        x1_1, x1_2 = x0, x0
        y1_1, y1_2 = y0 + n, y0 - n
    # General case
    else:
        ls_slope = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
        m = -1 / ls_slope

        x1_1 = x0 + n / math.sqrt(1 + m**2)
        y1_1 = y0 + m * (x1_1 - x0)
        x1_2 = x0 - n / math.sqrt(1 + m**2)
        y1_2 = y0 + m * (x1_2 - x0)

    # Rings formed by pt1, pt2 and coordinates computed above
    lr1 = LinearRing([pt1, pt2, (x1_1, y1_1)])
    lr2 = LinearRing([pt1, pt2, (x1_2, y1_2)])

    # If ring is counter clockwise
    if lr1.is_ccw:
        x_ccw, y_ccw = x1_1, y1_1
        x_cw, y_cw = x1_2, y1_2
    else:
        x_ccw, y_ccw = x1_2, y1_2
        x_cw, y_cw = x1_1, y1_1

    # If offset is positive, coordinate on the left
    if n > 0:
        x1, y1 = x_ccw, y_ccw
    # Else, coordinate on the right
    else:
        x1, y1 = x_cw, y_cw

    return x1, y1

def viz_trajectory(
        xy_pred: np.ndarray,
        centerline: np.ndarray
        ):
    
    plt.figure(figsize=(8, 7))
    visualize_centerline(centerline)
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "-",
        color="#d33e4c",
        alpha=1,
        linewidth=3,
        zorder=15,
    )

    final_x = xy[-1, 0]
    final_y = xy[-1, 1]

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
    plt.show()

def viz_best_worst_case(
        output_dic,
        dis_list
        ):
    ind = np.argsort(dis_list)
    viz_trajectory(output_dic[ind[1000]][0],output_dic[ind[1000]][1])
    
if __name__ == "__main__":
    data_dir = 'features/features_test.pkl'
    model_dir = 'models/LSTM_2.pth.tar'
    batch_size = 5
    mode = "test"
    
    cuda = torch.cuda.is_available()
    if cuda:
        device = torch.device('cuda')
        model = torch.load(model_dir)
    else:
        device = torch.device('cpu')
        model = torch.load(model_dir,map_location=torch.device('cpu'))
    
    start = time.time()
    # load best model
    encoder = LSTMEncoder()
    decoder = LSTMDecoder()
    
    encoder.load_state_dict(model['encoder_state_dict'])
    encoder.eval()
    decoder.load_state_dict(model['decoder_state_dict'])
    decoder.eval()
    
    # load test data
    if mode == 'test':
        data_dict = load_and_preprocess(data_dir,TEST_FEATURE_FORMAT)
    else:
        data_dict = load_and_preprocess(data_dir,FEATURE_FORMAT)
    dataset = data_loader(data_dict)
    test_loader = DataLoader(dataset,batch_size=batch_size,collate_fn=my_collate)
    
    loss_function = nn.MSELoss()
    loss_list = []
    dis_list = []
    ADE_list = []
    
#    output_dict = []
    
    for batch_id,(seq_id,input_data,target,ground_truth,cl_list,delta_ref) in enumerate(test_loader):
#        print(input_data.shape,target.shape,ground_truth.shape,len(cl_list),delta_ref.shape)
        for cl_idx in range(input_data.shape[0]):
            if len(cl_list[cl_idx].shape) != 2:
                input_data = np.delete(input_data,cl_idx,0)
                target = np.delete(target,cl_idx,0)
                ground_truth = np.delete(ground_truth,cl_idx,0)
                del cl_list[cl_idx]
                delta_ref = np.delete(delta_ref,cl_idx,0)
        if batch_id%100 == 0:
            print(f"\r Evaluating {batch_id*batch_size}/{len(test_loader)*batch_size}",end="")
        input_data = input_data.to(device)
        target = target.to(device)

        #Define encoder arguments
        sample_size = input_data.shape[0]
        input_size = input_data.shape[1]
        output_size = ground_truth.shape[1]

        #Initialize encoder hidden state
        hx = torch.zeros(sample_size, encoder.hidden_size).to(device)
        cx = torch.zeros(sample_size, encoder.hidden_size).to(device)
        encoder_hidden = (hx,cx)

        #Encoder observed trajectory
        for encoder_idx in range(input_size):
            encoder_input = input_data[:, encoder_idx, :]
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
        loss_list.append(loss.detach().numpy())
        
        nt_abs = get_nt_abs_values(
                decoder_outputs.detach().cpu().clone().numpy(),
                delta_ref
                )
        xy_abs = get_xy_from_nt_seq(nt_abs,cl_list)
        
#        for seq in range(len(seq_id)):
#            output_dict.append([xy_abs[seq,:,:],cl_list[seq]])
        
        dis = np.linalg.norm((xy_abs[:,-1,:]-ground_truth[:,-1,:]),axis=1)
        ade = np.sum(np.linalg.norm((xy_abs-ground_truth),axis=2),axis=1)/output_size
        dis_list = np.concatenate((dis_list,dis))
        ADE_list = np.concatenate((ADE_list,ade))

    #Average the loss (for all batches)
    val_loss = (sum(loss_list)/len(loss_list))
    FDE = sum(dis_list)/len(dis_list)
    ADE = sum(ADE_list)/len(ADE_list)
    
    print(f"Average loss: {val_loss}, Average FDE: {FDE}, Average ADE: {ADE}")
    print(f"Total time: {time.time()-start}")
    
    fig1=plt.figure(1)
    plt.hist(dis_list,bins=20,range=(0,20))
    plt.title("Final Displacement Error distribution of test dataset")

    fig2=plt.figure(2)
    plt.hist(ADE_list,bins=20,range=(0,10))
    plt.title("Average Displacement Error distribution of test dataset")

    
#    generate_forecasting_h5(output_dict, 'com/')
#    viz_ind = np.random.randint(len(dis_list))
#    viz_best_worst_case(output_dict,dis_list)
#Average loss: 9.367685013698052, Average FDE: 4.204276200197748, Average ADE: 2.089169880761964