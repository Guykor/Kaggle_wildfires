import pandas as pd


class GeographicFeatures:
    """
    Handles geographic features
    """

    # todo: handle geographic features

    def __init__(self, df):
        self.df = df[self.get_input_cols()]

    def get_input_cols(self):
        """
        Geo features data
        """
        return ['STATE',
                'COUNTY',
                'OWNER_DESCR',
                'LATITUDE',
                'LONGITUDE']

    def get_features(self):
        """
        returns a DataFrame with the relevant categorical features
        """
        pd.concat([self.df, pd.get_dummies(self.df, columns=['STATE'], prefix='state')], axis=1)
        to_drop = ['STATE',
                   'COUNTY',
                   'OWNER_DESCR']
        return self.df.drop(columns=to_drop, errors='ignore')  # drop only existing labels
