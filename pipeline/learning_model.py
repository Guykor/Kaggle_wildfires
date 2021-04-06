import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, classification_report, precision_score
import xgboost as xgb
from lightgbm import LGBMClassifier
import lightgbm
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from optuna.visualization.matplotlib import *
from sklearn.neural_network import MLPClassifier
from Logger import Logger
from pre_processor import PreProccessor

import optuna

# def create_confusion_matrix_and_analyze(pred, test_y):
# unique_label = np.unique([test_y, pred])
# conf_matrix = pd.DataFrame(
#     confusion_matrix(test_y, pred, labels=unique_label),
#     index=['true:{:}'.format(x) for x in unique_label],
#     columns=['pred:{:}'.format(x) for x in unique_label])
# print(conf_matrix)


logger = Logger()


def print_importance_of_rf_features(X_train, model_instance):
    '''
    this function receives the train set (X and y) and a model instance of Random Forest and prints
    the 20 most important features, by their ranking and with their weights
    :param X_train:
    :param model_instance:
    :param y_train:
    :return:
    '''
    feature_rank = np.argsort(model_instance.feature_importances_)[::-1]
    print("10 most important features of this model are: ")
    for i, index in enumerate(feature_rank[:10]):
        print('{0:d}. {1:s} ({2:2.2f})'.format(i + 1, X_train.columns[index],
                                               model_instance.feature_importances_[index]))


def learn_base_model_with_random_search(X, y, model, model_name):
    print("Started learning a base model")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=137,
                                                        stratify=y)
    print("started fit")
    search = model.fit(X_train, y_train)
    print("finished fit")
    print(model.__class__)
    if model_name == "RandomForestClassifier":
        print_importance_of_rf_features(X_train, model)
    if model_name == "CV":
        model = search.best_estimator_
        print(search.best_params_)
        print(search.cv_results_)

    evaluate_model(X_test, model, y_test, y.unique(), name_of_log_file='best_mode_of_random_search')


# TODO to save: parameters, features used, accuracy/other metrics

# TODO preprocessing: handle nans (e.g in CONT_DATE)


def evaluate_model(X_test, clf, y_test, labels_names, name_of_log_file='no_name', notes='no_note'):
    print("Started evaluating model")
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_confusion_matrix(clf, X_test, y_test, normalize='true', xticks_rotation='vertical', ax=ax)
    plt.title('Normalized by Rows (recall)')
    plt.show()
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_confusion_matrix(clf, X_test, y_test, normalize='pred', xticks_rotation='vertical', ax=ax)
    plt.title('Normalized by Columns (precision)')
    plt.show()
    y_predictions = clf.predict(X_test)
    unique_predictions = np.unique(y_predictions)
    print("rate of unique labels predicted is: ")
    print(len(unique_predictions) / len(labels_names))
    print("unique predictions are: ")
    print(unique_predictions)
    print(classification_report(y_test, y_predictions, target_names=labels_names))

    if type(clf).__name__ == 'LGBMClassifier':
        print_importance_of_rf_features(X_test, clf)

    precision_score_of_model = precision_score(y_test, y_predictions, average='weighted')
    logger.log_run(name_of_log_file, clf, precision_score_of_model, confusion_matrix=0, notes=notes)


# def read_file():
# 	size_dict = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
# 	print("Started reading file from disk")
# 	if not os.path.exists('fires.pickle'):
# 		conn = sqlite3.connect("FPA_FOD_20170508.sqlite")
# 		fires_tbl = pd.read_sql("select * from fires", conn)
# 		with open('fires.pickle', 'wb') as f:
# 			pickle.dump(fires_tbl, f)
# 	else:
# 		with open('fires.pickle', 'rb') as f:
# 			fires_tbl = pickle.load(f)
# 	fires_tbl['DISCOVERY_DATE'] = pd.to_datetime(
# 		fires_tbl['DISCOVERY_DATE'] - pd.Timestamp(0).to_julian_date(),
# 		unit='D')
# 	fires_tbl['CONT_DATE'] = pd.to_datetime(fires_tbl['CONT_DATE'] - pd.Timestamp(0).to_julian_date(),
# 											unit='D')
# 	fires_tbl['controlled_month'] = fires_tbl['CONT_DATE'].apply(lambda x: x.month)
#
# 	features = ['FIRE_YEAR', 'FIRE_SIZE', 'FIRE_SIZE_CLASS', 'LATITUDE', 'LONGITUDE', 'STAT_CAUSE_DESCR',
# 				'controlled_month', 'STATE']
# 	# features = ['FIRE_YEAR', 'FIRE_SIZE', 'LATITUDE', 'LONGITUDE',
# 	#             'DISCOVERY_DATE', 'CONT_DATE', 'STAT_CAUSE_DESCR']
# 	# features = ['STATE', 'COUNTY', 'FIRE_YEAR', 'FIRE_SIZE', 'FIRE_SIZE_CLASS', 'LATITUDE', 'LONGITUDE',
# 	#      'DISCOVERY_DATE', 'CONT_DATE', 'STAT_CAUSE_DESCR']
# 	fires_tbl = fires_tbl[features]
#
# 	fires_tbl["FIRE_SIZE_CLASS"] = fires_tbl["FIRE_SIZE_CLASS"].replace(size_dict)
#
# 	columns_to_dummies = ['FIRE_YEAR', 'STATE']
# 	fires_tbl = pd.get_dummies(fires_tbl, columns=columns_to_dummies)
#
# 	# TODO now changing nans to average, to change later
# 	fires_tbl.fillna(fires_tbl.median(), inplace=True)
# 	print("Finished reading file and pre processing it")
# 	return fires_tbl


def create_train_test(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                      test_size=0.2,
                                                      random_state=137,
                                                      stratify=y)
    return X_train, X_val, y_train, y_val


class Objective(object):
    def __init__(self, X, y):
        X_train, X_val, y_train, y_val = create_train_test(X, y)
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val

    def __call__(self, trial):
        global lgbm_model_of_trial
        n_estimators = trial.suggest_int('n_estimators', 1, 200)
        max_depth = trial.suggest_int('max_depth', 2, 100)
        num_leaves = trial.suggest_int('num_leaves', 2, 200)
        balanced_or_not = trial.suggest_categorical('balanced_or_not', ['balanced', None])
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        # l2 regularization
        reg_lambda = trial.suggest_uniform('reg_lambda', 0.0, 20.0)

        lgbm_model_of_trial = LGBMClassifier(class_weight=balanced_or_not,
                                             n_estimators=n_estimators,
                                             max_depth=max_depth,
                                             num_leaves=num_leaves,
                                             learning_rate=learning_rate,
                                             reg_lambda=reg_lambda)

        lgbm_model_of_trial.fit(self.X_train, self.y_train)
        y_pred = lgbm_model_of_trial.predict(self.X_val)

        # maximizing weighted average precision
        precision_error = 1 - precision_score(self.y_val, y_pred, average='weighted')
        return precision_error


def callback(study, trial):
    global best_model_of_optuna
    if study.best_trial == trial:
        best_model_of_optuna = lgbm_model_of_trial


def learn_model_smart_search():
    """
    This function learns LGBM model using optuna optimization and logs the best model
    :return:
    """

    prep = PreProccessor('FPA_FOD_20170508.sqlite')
    X, y = prep.get_features_data()

    study = optuna.create_study()  # Create a new study.
    # study.optimize(Objective(read_file()), n_trials=50, callbacks=[callback])  # Invoke optimization of the objective function.
    X_train, X_test, y_train, y_test = create_train_test(X, y)
    study.optimize(Objective(X, y), n_trials=15,
                   callbacks=[callback])  # Invoke optimization of the objective function.
    print("best model is")
    print(best_model_of_optuna)
    print("Best parameters are: ")
    print(study.best_params)
    print("Best value of objective is: ")
    print(study.best_value)
    logger.set_features(X)
    evaluate_model(X_test,
                   best_model_of_optuna,
                   y_test,
                   y_train.unique(),
                   name_of_log_file='optuna_best_model',
                   notes='no_note')

    # plots_for_optuna_studies(study)
    print("Finished learn_model_smart_search")


def plots_for_optuna_studies(study):
    """
    This function plots the summarized results of the studies
    :param study:
    :return:
    """
    plot_parallel_coordinate(study)
    plt.show()
    plot_optimization_history(study)
    plt.show()
    plot_edf(study)
    plt.show()
    plot_intermediate_values(study)
    plt.show()


def learn_nn():
    """
    Learning MLP model
    :return:
    """
    print("Stated learning NN")
    prep = PreProccessor('FPA_FOD_20170508.sqlite')
    X, y = prep.get_features_data()
    X_train, X_test, y_train, y_test = create_train_test(X, y)
    print("Started fit of NN")
    mlp_clf = MLPClassifier(random_state=137, hidden_layer_sizes=(30, 30, 30),
                            learning_rate='adaptive').fit(
        X_train, y_train)
    print("Finished fit of NN")
    evaluate_model(X_test, mlp_clf, y_test, y_train.unique(), name_of_log_file='nn_model')


def main():
    learn_model_smart_search()


# learn_nn()
# learn_base_model_random_search()
# learn_specific_lgbm()
#

def learn_specific_lgbm():
    """
    This function is for learning LGBM model with specific parameters
    :return:
    """

    prep = PreProccessor('FPA_FOD_20170508.sqlite')
    X, y = prep.get_features_data()

    print("Started learning specific lgbm")
    logger.set_features(X)
    X_train, X_test, y_train, y_test = create_train_test(X, y)

    if not os.path.exists('lgbm_test.pkl'):
        print("Learning from scratch")
        lgbm_model = LGBMClassifier(class_weight='balanced', max_depth=18, n_estimators=80,
                                    num_leaves=49)
        lgbm_model.fit(X_train, y_train)
        with open('lgbm_test.pkl', 'wb') as out_file:
            pickle.dump(lgbm_model, out_file)
        print("Finished learning")

    else:
        print("Reading from pickle")
        with open('lgbm_test.pkl', 'rb') as in_file:
            lgbm_model = pickle.load(in_file)
        print("Finished reading from pickle")

    evaluate_model(X_test, lgbm_model, y_test, y_train.unique(),
                   name_of_log_file='lgbm_with_specific_params')


def learn_base_model_random_search():
    """
    This is for learning base LGBM model
    :return:
    """
    prep = PreProccessor('FPA_FOD_20170508.sqlite')
    X, y = prep.get_features_data()
    # rf_clf = RandomForestClassifier(n_estimators=20, max_depth=4, class_weight='balanced')
    # xgb_model = xgb.XGBClassifier()
    lgbm_model = LGBMClassifier(class_weight='balanced')
    distributions = dict(n_estimators=randint(1, 150), max_depth=randint(2, 50),
                         num_leaves=randint(2, 100))
    clf = RandomizedSearchCV(lgbm_model, distributions, random_state=0, n_jobs=-1)
    learn_base_model_with_random_search(X, y, clf, "CV")


if __name__ == '__main__':
    main()
