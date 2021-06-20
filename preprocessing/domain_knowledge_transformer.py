import ray
import numpy as np
import pandas as pd #modin.pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DomainKnowledgeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,remove_original=True):
        super().__init__()
        self.remove_original=remove_original
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
            ### "Id" - feature (it's only index -remove)
        X_ = X_.drop(['Id'], axis=1)

        ### "MSSubClass" - feature (it's only index -remove)
        X_['_MSSubClass']=X_['MSSubClass'].astype(dtype='category')
        ####df['_MSSubClass'] = df['MSSubClass'].apply(lambda x: 'True' if x <= 53 else 'False')
        if self.remove_original:
            X_ = X_.drop(['MSSubClass'], axis=1)

            ### "MSZoning" - feature prepare for missing values
        X_['_MSZoning']=np.where(df['MSZoning'].isnull(),np.nan,df['MSZoning'].values)
        if self.remove_original:
            X_ = X_.drop(['MSZoning'], axis=1)

        ### "LotFrontage" - feature prepare for missing values
        X_['_LotFrontage']=np.where(df['LotFrontage'].isnull(),np.nan,df['LotFrontage'].values)
        if self.remove_original:
            X_ = X_.drop(['LotFrontage'], axis=1)

        return X_


if __name__ == '__main__':
    verbose=False
    #Using modin
    """
    try:
        ray.init()
    except:
        ray.shutdown()
        ray.init()
    """

    # make all dataframe columns visible
    pd.set_option('display.max_columns', None)

    df = pd.read_csv('/home/peterpirog/PycharmProjects/BostonHousesTune/preprocessing/train.csv')
    #df.to_excel('/home/peterpirog/PycharmProjects/BostonHousesTune/preprocessing/train.xlsx',
    #            sheet_name='X_train_data',
    #            index=False)
    #print(df.head())

    dkt=DomainKnowledgeTransformer()
    df_out=dkt.fit_transform(X=df)
    df_out.to_excel('/home/peterpirog/PycharmProjects/BostonHousesTune/preprocessing/output_data.xlsx',
                sheet_name='output_data',
                index=False)

    print(df_out.head(10))
    print(df_out.info())