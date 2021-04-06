import pandas as pd
import numpy as np
import sqlite3
import os

pd.options.mode.chained_assignment = None  # default='warn'

from date_features import DateFeatures
from geographic_features import GeographicFeatures
from size_features import SizeFeatures

ORIGINAL_COLS = ['OBJECTID', 'FOD_ID', 'FPA_ID',
                 'SOURCE_SYSTEM_TYPE', 'SOURCE_SYSTEM',
                 'NWCG_REPORTING_AGENCY', 'NWCG_REPORTING_UNIT_ID',
                 'NWCG_REPORTING_UNIT_NAME', 'SOURCE_REPORTING_UNIT',
                 'SOURCE_REPORTING_UNIT_NAME', 'LOCAL_FIRE_REPORT_ID',
                 'LOCAL_INCIDENT_ID', 'FIRE_CODE', 'FIRE_NAME',
                 'ICS_209_INCIDENT_NUMBER', 'ICS_209_NAME', 'MTBS_ID',
                 'MTBS_FIRE_NAME', 'COMPLEX_NAME', 'FIRE_YEAR', 'DISCOVERY_DATE', 'DISCOVERY_DOY',
                 'DISCOVERY_TIME', 'STAT_CAUSE_CODE', 'STAT_CAUSE_DESCR', 'CONT_DATE',
                 'CONT_DOY', 'CONT_TIME', 'FIRE_SIZE', 'FIRE_SIZE_CLASS', 'LATITUDE',
                 'LONGITUDE', 'OWNER_CODE', 'OWNER_DESCR', 'STATE', 'COUNTY',
                 'FIPS_CODE', 'FIPS_NAME', 'Shape']

IRRELEVANT_FEATURES = ['OBJECTID',
                       'FOD_ID',
                       'FPA_ID',
                       'SOURCE_SYSTEM_TYPE',
                       'SOURCE_SYSTEM',
                       'NWCG_REPORTING_AGENCY',
                       'NWCG_REPORTING_UNIT_ID',
                       'NWCG_REPORTING_UNIT_NAME',
                       'SOURCE_REPORTING_UNIT',
                       'SOURCE_REPORTING_UNIT_NAME',
                       'LOCAL_FIRE_REPORT_ID',
                       'LOCAL_INCIDENT_ID',
                       'FIRE_CODE',
                       'FIRE_NAME',
                       'ICS_209_INCIDENT_NUMBER',
                       'ICS_209_NAME',
                       'MTBS_ID',
                       'MTBS_FIRE_NAME',
                       'COMPLEX_NAME',
                       'STAT_CAUSE_CODE',
                       'OWNER_CODE',
                       'FIPS_CODE',
                       'FIPS_NAME',
                       'Shape']

FEATURE_COLS = ['FIRE_YEAR',
                'DISCOVERY_DATE',
                'DISCOVERY_DOY',
                'DISCOVERY_TIME',
                'CONT_DATE',
                'CONT_DOY',
                'CONT_TIME',
                'FIRE_SIZE',
                'LATITUDE',
                'LONGITUDE']

CATEGORICAL = ['FIRE_SIZE_CLASS',
               'OWNER_DESCR',
               'STATE',
               'COUNTY']

DUMMIES_COLS = []

LABEL_COL = 'STAT_CAUSE_DESCR'

RAW_NAME_PKL_FILE = 'fires.pickle'
FEATURES_NAME_PKL_FILE = 'fires_features.pickle'


class PreProccessor:
    """
    Preproccesses the data
    """

    def __init__(self, db_path):
        """
        :param db_path: A path to the fires DB
        :type db_path:
        """
        db_path = os.path.abspath(db_path)
        self.client = sqlite3.connect(db_path)
        self.fire_id = 'OBJECTID'
        self.fires_table = None
        if not os.path.exists(FEATURES_NAME_PKL_FILE):
            self.load_raw_data()

    def get_features_data(self):
        """
        :return: returns X, y: a clean features table and response y
        :rtype:
        """
        if os.path.exists(FEATURES_NAME_PKL_FILE):
            print("Preprocessor: uploading feature matrix from pickle....")
            df = pd.read_pickle(FEATURES_NAME_PKL_FILE)
            y = df.pop(LABEL_COL)
            print("Preprocessor: finished....")
            print("Preprocessor: Features are: ")
            print(list(df.columns))
            return df, y

        print("Preprocessor: Baking Features...")
        features_dfs = []
        date = DateFeatures(self.fires_table)
        features_dfs.append(date.get_features())

        size = SizeFeatures(self.fires_table)
        features_dfs.append(size.get_features())

        # todo: handle geographic features
        geographic = GeographicFeatures(self.fires_table)
        features_dfs.append(geographic.get_features())
        print("Preprocessor: Complete")
        X = pd.concat(features_dfs, axis=1)
        y = self.fires_table[LABEL_COL]
        save = pd.concat([X, y], axis=1)
        save.to_pickle(FEATURES_NAME_PKL_FILE)
        print("Preprocessor: Features are: ")
        print(list(X.columns))
        return X, y

    def load_raw_data(self):
        if os.path.exists(RAW_NAME_PKL_FILE):
            print("Preprocessor: Read from pickle...")
            self.fires_table = pd.read_pickle(RAW_NAME_PKL_FILE)
            print("Preprocessor: Complete")
        else:
            print("Preprocessor: Read from DB...")
            self.fires_table = pd.read_sql('select * from fires', self.client)
            self.fires_table.to_pickle(RAW_NAME_PKL_FILE)
            print("Preprocessor: Complete")
        self.fires_table = self.fires_table.set_index('OBJECTID')
