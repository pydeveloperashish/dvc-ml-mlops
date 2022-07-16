from email import contentmanager
import yaml
import os
import json

def read_yaml(path_to_yaml: str) -> dict:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
    return content


def create_directory(dirs: list):
    for dir_path in dirs:
       os.makedirs(dir_path, exist_ok=True) 
       print(f"Directory is created at {dir_path}")
       

def save_local_df(data, data_path):
    data.to_csv(data_path, index = False)
    print(f"Data is saved at {data_path}")
    
    
def save_reports(report: dict, report_path: str):
    with open(report_path, 'w') as f:
        json.dump(report, f, indent = 4)
    print(f"Reports are saved at {report_path}")    