import pandas as pd

class SizeFeatures:
    """
    Handles categorical data
    """

    def __init__(self, df):
        self.df = df[self.get_input_cols()]

    def get_input_cols(self):
        """
        Date features data
        """
        return ['FIRE_SIZE','FIRE_SIZE_CLASS']

    def get_features(self):
        """
        returns a DataFrame with the relevant categorical features
        """
        self.size_class()
        self.size_linear()
        return self.df.drop(columns=self.get_input_cols(), errors='ignore') # drop only existing
        # labels

    def size_linear(self):
        self.df['fire_size'] = self.df['FIRE_SIZE']

    def size_class(self):
        """
        add dummies to data
        """
        columns_to_dummies = ['FIRE_SIZE_CLASS']
        self.df = pd.get_dummies(self.df, columns=columns_to_dummies, prefix='size_class')