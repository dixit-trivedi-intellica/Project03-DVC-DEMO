from ast import literal_eval
from src.utils.training_common_utils import *
from src.utils.common_utils import *
import argparse
## import libraries
import gc
import ast
import logging
import math
import os
import torch
from ast import literal_eval
from collections import Counter
import pandas as pd
import numpy as np
from shutil import rmtree, copyfile
from multiprocesspandas import applyparallel
from tqdm import trange
# from .email_functions_training import email_send
from transformers import AdamW
from transformers import XLNetModel
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import accuracy_score
from transformers import AdamW, XLNetTokenizer
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import datetime 
import nltk
import contractions
from nltk.util import ngrams
import string
from flask import abort, Response
from spellchecker import SpellChecker
import json
nltk.download('averaged_perceptron_tagger')
spell = SpellChecker()

def training(config_path):
    
    config = read_params(config_path)
    artifacts = config["artifacts"]  
    artifacts_dir = config["artifacts"]["artifacts_dir"] 
    train_data_path = artifacts["preprocess_data"]["train_path"]
    
    model_dir = artifacts["model_dir"]
    model_path = artifacts["model_path"]
    
    label_cols = artifacts["train_params"]["labels_cols_34"]
    comm_col = artifacts["train_params"]["comment_col"]
    tgt_col = artifacts["train_params"]["tgt_col"]
    
    batch_size = artifacts["train_params"]["batch_size"]
    
    create_dir(dirs=[artifacts_dir, model_dir])
    
    df = pd.read_csv(train_data_path, sep=",")
    print(df)
    
    df['labels'] = df['labels'].apply(lambda x: literal_eval(x))

    for i in label_cols:
        df[i] = 0
    for i in label_cols:
        df[i] = df[tgt_col].apply(lambda x: 1 if i in x else 0)

    df.drop([tgt_col], axis=1, inplace=True)

    data = df[[comm_col]]
    data[label_cols] = df[label_cols]

    X = data[comm_col]
    y = data[label_cols]

    train_data, valid_data, train_label, valid_label = train_test_split(X, y, test_size=0.1, random_state=0)

    # prepare_data function returns dataframe with statement, label, all dummies column, features and masks
    training = prepare_data(train_data, train_label)
    valid = prepare_data(valid_data, valid_label)

    X_train = training["features"].values.tolist()
    X_valid = valid["features"].values.tolist()

    train_masks = training["masks"].values.tolist()
    valid_masks = valid["masks"].values.tolist()

    Y_train = train_label.values.tolist()
    Y_valid = valid_label.values.tolist()

    X_train = torch.tensor(X_train)
    X_valid = torch.tensor(X_valid)

    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    Y_valid = torch.tensor(Y_valid, dtype=torch.float32)

    train_masks = torch.tensor(train_masks, dtype=torch.long)
    valid_masks = torch.tensor(valid_masks, dtype=torch.long)

    
    # Create an iterator of our data with torch DataLoader. This helps save on 
    # memory during training because, unlike a for loop, 
    # with an iterator the entire dataset does not need to be loaded into memory
    train_data = TensorDataset(X_train, train_masks, Y_train)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data,\
                                  sampler=train_sampler,\
                                  batch_size=batch_size)

    validation_data = TensorDataset(X_valid, valid_masks, Y_valid)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data,\
                                       sampler=validation_sampler,\
                                       batch_size=batch_size)

    current_time = datetime.datetime.now()
    curr_time_stamp =  str(current_time.year) + str(current_time.month) + "_" + str(current_time.day) + "_" + str(current_time.hour) + "_" + str(current_time.minute) + "_"  + str(current_time.second)
    
    model = XLNetForMultiLabelSequenceClassification(num_labels=len(Y_train[0]))
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01, correct_bias=False)

    model_save_path = model_path
    model, train_loss_set, valid_loss_set = train(model=model, num_epochs=5, optimizer=optimizer, train_dataloader=train_dataloader,
          valid_dataloader=validation_dataloader, model_save_path=model_save_path, device="cuda")
    gc.collect()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    parse_args = args.parse_args()
    
    try:
        data = training(config_path=parse_args.config)
    except Exception as e:
        raise e 
    