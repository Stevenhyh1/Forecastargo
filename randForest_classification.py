# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 17:24:34 2019

@author: Quanwei
"""
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score

#Directories
train_dir = 'features/classfeatuer/features_class.pkl'
val_dir = 'features/classfeatuer/features_class.pkl'
save_dir = 'models'

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

def main():
    """
    step1:load data
    """
    data_train_dict=load_and_preprocess(train_dir)
    data_val_dict=load_and_preprocess(val_dir)
    
    data_train_inputs=np.array(data_train_dict["input"])
    data_train_outputs=np.array(data_train_dict["output"])
    data_val_inputs=np.array(data_val_dict["input"])
    data_val_outputs=np.array(data_val_dict["output"])
    print(data_train_inputs.shape)
    """
    step2:data processing
    """
    num_sample_train=data_train_inputs.shape[0]
    num_sample_val=data_val_inputs.shape[0]
    num_timestep_train=data_train_inputs.shape[1]
    num_timestep_val=data_val_inputs.shape[1]
#    num_features=data_train_inputs.shape[2]
    """
    train_x and train_y
    """
    for i in range(num_sample_train):
        if i==0:
            train_x=data_train_inputs[i,:,:]
        else:
            train_x=np.vstack((train_x,data_train_inputs[i,:,:]))
        for j in range(num_timestep_train):
            if j==0 and i==0:
                train_y=data_train_outputs[i]
            else:
                train_y=np.vstack((train_y,data_train_outputs[i]))
    """
    val_x and val_y
    """
    for i in range(num_sample_val):
        if i==0:
            val_x=data_val_inputs[i,:,:]
        else:
            val_x=np.vstack((val_x,data_val_inputs[i,:,:]))
        for j in range(num_timestep_val):
            if j==0 and i==0:
                val_y=data_val_outputs[i]
            else:
                val_y=np.vstack((val_y,data_val_outputs[i]))
#    print(train_x.shape)
#    print(train_y.shape)
#    print(val_x.shape)
#    print(val_y.shape)
    
    """
    step2:create random forest classifier
    """
    rf=RandomForestClassifier(n_jobs=-1,random_state=0)
    
    """
    step3:train the model
    """
    print("Training begins...")
    train_start=time.time()
    rf.fit(train_x,train_y)
    train_end=time.time()
    print(f"Train completed in {train_end-train_start} s")
    """
    step4:validate the model
    """
    print("Validation begins...")
    val_start=time.time()
    output_predict=rf.predict(val_x)
    val_end=time.time()
    acc=accuracy_score(val_y,output_predict)
    print(f"Validation completed in {val_end-val_start} s, the accuracy is {acc}")
if __name__=="__main__":
    main()