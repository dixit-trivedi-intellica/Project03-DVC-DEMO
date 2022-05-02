import argparse
import pandas as pd
import os
import contractions
import string
from multiprocesspandas import applyparallel
from src.utils.common_utils import read_params, create_dir, save_local_df
from spellchecker import SpellChecker

spell = SpellChecker()

def correct_spellings(text):
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            if len(word)<5:
                corrected_text.append(word)
                continue
            else:
                corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    list1 = " ".join(list(corrected_text))
    return list1

def pre_process(data, column_name):
    new_column = 'pre_process_' + str(column_name).lower()
    # convert columns into lower case 
    for col in data.columns:
        data.rename({col: col.lower()}, inplace=True, axis=1)
    print("cols", data.columns)
    # convert tweets to lower case
    data[new_column] = data[column_name].str.lower()
    # contractions
    data[new_column] = data[new_column].apply(lambda x: contractions.fix(x))
    # remove urls
    data[new_column] = data[new_column].str.replace(r'(https|http)?:\/(\w|\.|\/|\?|\=|\&|\%)*\b', ' ')
    data[new_column] = data[new_column].str.replace(r'www\.\S+\.com', ' ')
    # remove html tags
    data[new_column] = data[new_column].str.replace(r'<.*?>', ' ')
    # remove \n\t by space
    data[new_column] = data[new_column].str.replace(r'[\n|\r|\t]', ' ')
    # remove punctuation
    data[new_column] = data[new_column].str.replace('[{}]'.format(string.punctuation), ' ')
    # remove special symbols
    data[new_column].str.replace(r'[^a-z]', '')
    # this is for correct spell
    data[new_column] = data[new_column].apply_parallel(lambda x: correct_spellings(x))
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    data.reset_index(inplace=True)
    return data

def clean_data(config_path):
    config = read_params(config_path)
    artifacts = config["artifacts"]  
    artifacts_dir = config["artifacts"]["artifacts_dir"] 
    raw_local_data = artifacts["raw_local_data"]
    preprocess_data = artifacts["preprocess_data"]
    preprocess_data_dir = preprocess_data["processed_data_dir"]
    train_data_path = preprocess_data["train_path"]
    data_path = artifacts["raw_local_data"]
    create_dir(dirs=[artifacts_dir, preprocess_data_dir])
        
    df = pd.read_csv(data_path, sep=",")
    
    pre_processed_df = pre_process(df, 'text') 

    save_local_df(pre_processed_df, train_data_path)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    parse_args = args.parse_args()
    
    try:
        data = clean_data(config_path=parse_args.config)
    except Exception as e:
        raise e 
    