import pandas as pd

INPUT_COLS = ['FIRE_YEAR',
              'DISCOVERY_DATE',
              'DISCOVERY_DOY',
              'DISCOVERY_TIME',
              'CONT_DATE',
              'CONT_DOY',
              'CONT_TIME']


class DateFeatures:
    """
    Handles Date features
    """

    def __init__(self, df):
        self.df = df[self.get_input_cols()]

    def get_input_cols(self):
        """
        Date features data
        """
        return ['FIRE_YEAR',
                'DISCOVERY_DATE',
                'DISCOVERY_DOY',
                'DISCOVERY_TIME',
                'CONT_DATE',
                'CONT_DOY',
                'CONT_TIME',
                'FIRE_SIZE_CLASS']

    def get_features(self):
        """
        returns a DataFrame with the relevant date features
        """
        # Fix Discovery Date Features
        self.preprocess_dates()
        self.time_to_control_fire()
        self.discover_month()
        self.year()
        return self.df.drop(columns=self.get_input_cols(), errors='ignore') # drop only existing
        # labels

    def format_date(self, series):
        return pd.to_datetime(series - pd.Timestamp(0).to_julian_date(), unit='D')

    def preprocess_dates(self):
        self.df['DISCOVERY_DATE'] = self.format_date(self.df['DISCOVERY_DATE'])
        self.df['CONT_DATE'] = self.format_date(self.df['CONT_DATE'])

    def time_to_control_fire(self):
        """Imputation with fire size class group mean."""
        self.df['days_to_control'] = self.df.CONT_DATE - self.df.DISCOVERY_DATE
        self.df['days_to_control'] = self.df.groupby("FIRE_SIZE_CLASS").transform(lambda x: x.fillna(x.mean()))
        self.df['days_to_control'] = self.df['days_to_control'].astype(int)

    def discover_month(self):
        self.df['discover_month'] = self.df['DISCOVERY_DATE'].apply(lambda x: x.month)
        self.df = pd.get_dummies(self.df, columns=['discover_month'], prefix='disco_month')

    def year(self):
        self.df = pd.get_dummies(self.df, columns=['FIRE_YEAR'], prefix='year')

