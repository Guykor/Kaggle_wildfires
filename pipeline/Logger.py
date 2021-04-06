
import pandas as pd
import json
import os
import pickle

class Logger:
    def __init__(self):
        self.features = False

    def set_features(self, df):
        """
        :param df: a dataframe
        :return:
        """
        self.features = df.columns


    def log_run(self, log_name, model, precision_score, confusion_matrix=0,
                  cls_report="", notes=""):

        model_name = type(model).__name__
        params = model.get_params()

        log = {}

        log['model_name'] = model_name
        log['log_name'] = log_name
        log['params'] = params
        log['features'] = list(self.features)
        log['precision_score'] = precision_score
        log['confusion_matrix'] = confusion_matrix
        log['classification_report'] = cls_report
        log['notes'] = notes

        dir_path = "logs/" + log_name + "/"

        try:
            os.mkdir(dir_path)
        except OSError:
            print("couldn't create log directory")

        json_path = dir_path + "details.json"
        with open(json_path, "w+") as file:
            json_obj = json.dumps(log, indent=4)
            file.write(json_obj)
            print("logged model")
            file.close()

        model_path = dir_path + "model.pkl"
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)


def compare_models():
    dir = "logs/"

    for filename in os.listdir(dir):
        with open(dir + filename, 'r') as f_log:
            log = json.load(f_log)
            model_name = log['model_name']
            pres_err = log['precision_error']
            print(model_name)
            print(pres_err)
            print("*")












