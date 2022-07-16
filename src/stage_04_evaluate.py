from src.utils.all_utils import create_directory, read_yaml, save_reports
import argparse
import os
import pandas as pd
import joblib
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()


def evaluate(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)
        
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    split_data_dir = config["artifacts"]["split_data_dir"]    
    test_data_filename = config["artifacts"]["test"]
    
    test_data_path = os.path.join(artifacts_dir, split_data_dir, test_data_filename)

    test_data = pd.read_csv(test_data_path)
    
    test_y = test_data['HeartDisease']
    test_x = test_data.drop('HeartDisease', axis = 1)
    
    test_x['Sex'] = label_encoder.fit_transform(test_x['Sex'])
    test_x['ChestPainType'] = label_encoder.fit_transform(test_x['ChestPainType'])
    test_x['RestingECG'] = label_encoder.fit_transform(test_x['RestingECG'])
    test_x['ExerciseAngina'] = label_encoder.fit_transform(test_x['ExerciseAngina'])
    test_x['ST_Slope'] = label_encoder.fit_transform(test_x['ST_Slope'])
    
    
    model_dir = config["artifacts"]["model_dir"]
    model_filename_path = config["artifacts"]["model_filename"]
    model_path = os.path.join(artifacts_dir, model_dir, model_filename_path)
    
    rf = joblib.load(model_path)
    predictions = rf.predict(test_x)
    accuracy = accuracy_score(predictions, test_y)
    print("accuracy:", accuracy * 100)
    
    reports_dir = config["artifacts"]["reports_dir"]
    reports_dir_path = os.path.join(artifacts_dir, reports_dir)
    create_directory([reports_dir_path])
    
    scores_file = config["artifacts"]["scores"]
    scores_filepath = os.path.join(reports_dir_path, scores_file)
    
    scores = {
        "accuracy": accuracy
    }
    
    save_reports(scores, scores_filepath)
    

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default = "config/config.yaml")
    args.add_argument("--params", "-p", default = "params.yaml")
    parsed_args = args.parse_args()
    evaluate(config_path = parsed_args.config, params_path = parsed_args.params)
    
     
    