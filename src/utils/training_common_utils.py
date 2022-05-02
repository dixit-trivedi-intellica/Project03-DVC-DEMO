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

## split_training_data
def split_training_data(X,y):
    """ function for split data into train and validation data """

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    
    for train_index, val_index in msss.split(X, y):
        print(len(train_index), len(val_index))
        train_data, valid_data = X[train_index], X[val_index]
        train_label, valid_label = y[train_index], y[val_index]
        return train_data, valid_data, train_label, valid_label
    return None

## prepare_data
def prepare_data(data, labels):
    """ function for prepare data for the model """
    
    # preparing data
    dataframe = pd.DataFrame(data)
    dataframe.reset_index(drop=True, inplace=True)

#     # preparing labels
#     label_df = pd.DataFrame(label)
#     label_df.reset_index(drop=True, inplace=True)

#     # adding label column in dataframe
#     dataframe['label'] = label_df[label_df.columns.tolist()].apply(lambda x: ','.join(x[x.notnull()]), axis=1)
#     # now dataframe variable has two columns : statement and label

#     # generating dummies from label column
#     dataframe_dummies = dataframe['label'].str.get_dummies(sep=',')
#     dataframe[dataframe_dummies.columns.tolist()] = dataframe_dummies[dataframe_dummies.columns]
#     # after this dataframe variable has statement, label and dummies columns
    
    # generating features and masks for training
    data_text_list = dataframe["pre_process_text"].values
    data_input_ids, data_attention_masks = generate_features(data_text_list)
    dataframe["features"] = data_input_ids.tolist()
    dataframe["masks"] = data_attention_masks
    # now dataframe variable has statement, label, dummies columns, features and masks

    return dataframe

## generating features and masks for training
def generate_features(train_text_list):
    """ function for create input_ids and attention masks for train data """
    
    train_input_ids = tokenize_inputs(train_text_list, num_embeddings=250)
    train_attention_masks = create_attn_masks(train_input_ids)
    gc.collect()
    return train_input_ids, train_attention_masks

## tokenize the text, then truncate sequence to the desired length minus 2 for
def tokenize_inputs(text_list, num_embeddings=512):
    """
    Tokenizes the input text input into ids. Appends the appropriate special
    characters to the end of the text to denote end of sentence. Truncate or pad
    the appropriate sequence length.
    """
    # tokenize the text, then truncate sequence to the desired length minus 2 for
    # the 2 special characters
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
    tokenized_texts = list(map(lambda t: tokenizer.tokenize(t)[:num_embeddings-2], text_list))
    # convert tokenized text into numeric ids for the appropriate LM
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    # append special token "<s>" and </s> to end of sentence
    input_ids = [tokenizer.build_inputs_with_special_tokens(x) for x in input_ids]
    # pad sequences
    input_ids = pad_sequences(input_ids, maxlen=num_embeddings, dtype="long", truncating="post", padding="post")
    gc.collect()
    return input_ids

## generating features and masks for training
def create_attn_masks(input_ids):
    """
    Create attention masks to tell model whether attention should be applied to
    the input id tokens. Do not want to perform attention on padding tokens.
    """
    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
    return attention_masks

## get dataloader  
def get_dataloader(X_local, data_mask, Y_local, batch_size=16):
    """
    function for convert data_mask to tensor format, create dataset
    """
    data_mask = torch.tensor(data_mask, dtype=torch.long)
    data = TensorDataset(X_local, data_mask, Y_local)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    gc.collect()
    return dataloader

## XLnet class 
class XLNetForMultiLabelSequenceClassification(torch.nn.Module):
    """
    Define class of XLNetForMultiLabelSequenceClassification
    """

    def __init__(self, num_labels=2):
        """
        Constructor for store lables, initialize pre-trained model of xlnet
        """
        super(XLNetForMultiLabelSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
        self.classifier = torch.nn.Linear(768, num_labels)

        torch.nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, token_type_ids=None, \
                attention_mask=None, labels=None):
        # last hidden layer
        last_hidden_state = self.xlnet(input_ids=input_ids, \
                                       attention_mask=attention_mask, \
                                       token_type_ids=token_type_ids)
        # pool the outputs into a mean vector
        mean_last_hidden_state = self.pool_hidden_state(last_hidden_state)
        logits = self.classifier(mean_last_hidden_state)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), \
                            labels.view(-1, self.num_labels))
            return loss
        else:
            return logits

    def freeze_xlnet_decoder(self):
        """
        Freeze XLNet weight parameters. They will not be updated during training.
        """
        for param in self.xlnet.parameters():
            param.requires_grad = False

    def unfreeze_xlnet_decoder(self):
        """
        Unfreeze XLNet weight parameters. They will be updated during training.
        """
        for param in self.xlnet.parameters():
            param.requires_grad = True

    def pool_hidden_state(self, last_hidden_state):
        """
        Pool the output vectors into a single mean vector
        """
        last_hidden_state = last_hidden_state[0]
        mean_last_hidden_state = torch.mean(last_hidden_state, 1)
        return mean_last_hidden_state
    
## Train model
def train(model, num_epochs, \
          optimizer, \
          train_dataloader, valid_dataloader, \
          model_save_path, \
          train_loss_set=[], valid_loss_set=[], \
          lowest_eval_loss=None, start_epoch=0, \
          device="cuda"
          ):
    """
    Train the model and save the model with the lowest validation loss
    """

    model.to(device)

    # trange is a tqdm wrapper around the normal python range
    for i in trange(num_epochs, desc="Epoch"):

        actual_epoch = start_epoch + i

        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # Tracking variables
        tr_loss = 0
        num_train_samples = 0

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            loss = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            # store train loss
            tr_loss += loss.item()
            num_train_samples += b_labels.size(0)
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            # scheduler.step()

        # Update tracking variables
        epoch_train_loss = tr_loss / num_train_samples
        train_loss_set.append(epoch_train_loss)

        print("Train loss: {0}".format(epoch_train_loss))

        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Tracking variables
        eval_loss = 0
        num_eval_samples = 0

        # Evaluate data for one epoch
        for batch in valid_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients,
            # saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate validation loss
                loss = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                # store valid loss
                eval_loss += loss.item()
                num_eval_samples += b_labels.size(0)

        epoch_eval_loss = eval_loss / num_eval_samples
        valid_loss_set.append(epoch_eval_loss)

        print("Valid loss: {0}".format(epoch_eval_loss))

        if lowest_eval_loss == None:
            lowest_eval_loss = epoch_eval_loss
            # save model
            save_model(model, model_save_path, actual_epoch, \
                       lowest_eval_loss, train_loss_set, valid_loss_set)
        else:
            if epoch_eval_loss < lowest_eval_loss:
                lowest_eval_loss = epoch_eval_loss
                # save model
                save_model(model, model_save_path, actual_epoch, \
                           lowest_eval_loss, train_loss_set, valid_loss_set)
        print("\n")

    gc.collect()
    return model, train_loss_set, valid_loss_set


## save model to disk
def save_model(model, save_path, epochs, lowest_eval_loss, train_loss_hist, valid_loss_hist):
    """
    Save the model to the path directory provided
    """
    model_to_save = model.module if hasattr(model, 'module') else model
    checkpoint = {'epochs': epochs, \
                  'lowest_eval_loss': lowest_eval_loss, \
                  'state_dict': model_to_save.state_dict(), \
                  'train_loss_hist': train_loss_hist, \
                  'valid_loss_hist': valid_loss_hist
                  }
    torch.save(checkpoint, save_path)
    print("Saving model at epoch {0} with validation loss of {1}".format(epochs, \
                                                                       lowest_eval_loss))
    gc.collect()
    return
