from src.utils.all_utils import create_directory, read_yaml, save_local_df
import argparse
import os
import pandas as pd
import joblib
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()


def train(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)
        
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    split_data_dir = config["artifacts"]["split_data_dir"]    
    train_data_filename = config["artifacts"]["train"]
    
    train_data_path = os.path.join(artifacts_dir, split_data_dir, train_data_filename)

    train_data = pd.read_csv(train_data_path)
    
    train_y = train_data['HeartDisease']
    train_x = train_data.drop('HeartDisease', axis = 1)
    
    train_x['Sex'] = label_encoder.fit_transform(train_x['Sex'])
    train_x['ChestPainType'] = label_encoder.fit_transform(train_x['ChestPainType'])
    train_x['RestingECG'] = label_encoder.fit_transform(train_x['RestingECG'])
    train_x['ExerciseAngina'] = label_encoder.fit_transform(train_x['ExerciseAngina'])
    train_x['ST_Slope'] = label_encoder.fit_transform(train_x['ST_Slope'])
    
    rf = RandomForestClassifier()
    rf.fit(train_x, train_y)
    
    model_dir = config["artifacts"]["model_dir"]
    model_dir_path = os.path.join(artifacts_dir, model_dir)
    create_directory([model_dir_path])
    
    model_filename_path = config["artifacts"]["model_filename"]
    model_path = os.path.join(model_dir_path, model_filename_path)
    
    joblib.dump(rf, model_path)
    print("model is saved")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default = "config/config.yaml")
    args.add_argument("--params", "-p", default = "params.yaml")
    parsed_args = args.parse_args()
    train(config_path = parsed_args.config, params_path = parsed_args.params)
    
     
    