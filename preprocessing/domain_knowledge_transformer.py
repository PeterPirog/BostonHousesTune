import ray
import numpy as np
import pandas as pd  # modin.pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DomainKnowledgeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, remove_original=True):
        super().__init__()
        self.remove_original = remove_original

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        ### "Id" - feature (it's only index -remove)
        X_ = X_.drop(['Id'], axis=1)

        ### "MSSubClass" - feature (it's only index -remove)
        X_['_MSSubClass'] = X_['MSSubClass'].astype(dtype='category')
        ####df['_MSSubClass'] = df['MSSubClass'].apply(lambda x: 'True' if x <= 53 else 'False')
        if self.remove_original:
            X_ = X_.drop(['MSSubClass'], axis=1)

            ### "MSZoning" - feature prepare for missing values
        X_['_MSZoning'] = np.where(df['MSZoning'].isnull(), np.nan, df['MSZoning'].values)
        if self.remove_original:
            X_ = X_.drop(['MSZoning'], axis=1)

        ### "LotFrontage" - feature prepare for missing values
        X_['_LotFrontage'] = np.where(df['LotFrontage'].isnull(), np.nan, df['LotFrontage'].values)
        if self.remove_original:
            X_ = X_.drop(['LotFrontage'], axis=1)

        ### "LotArea" - feature prepare for missing values
        X_['_LotArea'] = np.where(df['LotArea'].isnull(), np.nan, df['LotArea'].values)
        if self.remove_original:
            X_ = X_.drop(['LotArea'], axis=1)

        ### "Street" - default value Paved =1, Gravel=0
        X_['_Street'] = np.where(df['Street'] == 'Grvl', 0, np.nan)
        X_['_Street'] = np.where(df['Street'] == 'Pave', 1, X_['_Street'].values)
        if self.remove_original:
            X_ = X_.drop(['Street'], axis=1)

        ### "Alley" - default value Paved =2, Gravel=1, None=0
        X_['_Alley'] = np.where(df['Alley'].isnull(), 0, np.nan)
        X_['_Alley'] = np.where(df['Alley'] == 'Grvl', 1, X_['_Alley'].values)
        X_['_Alley'] = np.where(df['Alley'] == 'Pave', 2, X_['_Alley'].values)
        if self.remove_original:
            X_ = X_.drop(['Alley'], axis=1)

        ### "Alley" - default value np.nan
        X_['_LotShape'] = np.nan  # np.where(df['LotShape'].isnull(), np.nan, df['LotShape'].values)
        X_['_LotShape'] = np.where(df['LotShape'] == 'Reg', 3, X_['_LotShape'].values)
        X_['_LotShape'] = np.where(df['LotShape'] == 'IR1', 2, X_['_LotShape'].values)
        X_['_LotShape'] = np.where(df['LotShape'] == 'IR2', 1, X_['_LotShape'].values)
        X_['_LotShape'] = np.where(df['LotShape'] == 'IR3', 0, X_['_LotShape'].values)
        if self.remove_original:
            X_ = X_.drop(['LotShape'], axis=1)

        ### "LandContour" - default value np.nan
        X_['_LandContour'] = np.where(df['LandContour'].isnull(), np.nan, df['LandContour'].values)
        X_['_LandContour_IsFlat'] = np.where(df['LandContour'] == 'Lvl', 1, 0)  # additional feature
        if self.remove_original:
            X_ = X_.drop(['LandContour'], axis=1)

        ### "Utilities" - default value np.nan
        X_['_Utilities'] = np.nan
        X_['_Utilities'] = np.where(df['Utilities'] == 'AllPub', 3, X_['_Utilities'].values)
        X_['_Utilities'] = np.where(df['Utilities'] == 'NoSewr', 2, X_['_Utilities'].values)
        X_['_Utilities'] = np.where(df['Utilities'] == 'NoSeWa', 1, X_['_Utilities'].values)
        X_['_Utilities'] = np.where(df['Utilities'] == 'ELO', 0, X_['_Utilities'].values)
        if self.remove_original:
            X_ = X_.drop(['Utilities'], axis=1)

        ### "Alley" - default value np.nan
        X_['_LotConfig'] = np.where(df['LotConfig'].isnull(), np.nan, df['LotConfig'].values)
        if self.remove_original:
            X_ = X_.drop(['LotConfig'], axis=1)

        ### "LandSlope" - default value np.nan
        X_['_LandSlope'] = np.nan
        X_['_LandSlope'] = np.where(df['LandSlope'] == 'Gtl', 2, X_['_LandSlope'].values)
        X_['_LandSlope'] = np.where(df['LandSlope'] == 'Mod', 1, X_['_LandSlope'].values)
        X_['_LandSlope'] = np.where(df['LandSlope'] == 'Sev', 0, X_['_LandSlope'].values)
        if self.remove_original:
            X_ = X_.drop(['LandSlope'], axis=1)

        ### "Alley" - default value np.nan
        X_['_Neighborhood'] = np.where(df['Neighborhood'].isnull(), np.nan, df['Neighborhood'].values)
        if self.remove_original:
            X_ = X_.drop(['Neighborhood'], axis=1)

        ### Combine Condition 1 and Condition 2 - default value 0
        X_['_Condition_Artery'] = np.where((df['Condition1'] == 'Artery') | (df['Condition2'] == 'Artery'), 1, 0)
        X_['_Condition_Feedr'] = np.where((df['Condition1'] == 'Feedr') | (df['Condition2'] == 'Feedr'), 1, 0)
        X_['_Condition_Norm'] = np.where((df['Condition1'] == 'Norm') | (df['Condition2'] == 'Norm'), 1, 0)
        X_['_Condition_RRNn'] = np.where((df['Condition1'] == 'RRNn') | (df['Condition2'] == 'RRNn'), 1, 0)
        X_['_Condition_RRAn'] = np.where((df['Condition1'] == 'RRAn') | (df['Condition2'] == 'RRAn'), 1, 0)
        X_['_Condition_PosN'] = np.where((df['Condition1'] == 'PosN') | (df['Condition2'] == 'PosN'), 1, 0)
        X_['_Condition_PosA'] = np.where((df['Condition1'] == 'PosA') | (df['Condition2'] == 'PosA'), 1, 0)
        X_['_Condition_RRNe'] = np.where((df['Condition1'] == 'RRNe') | (df['Condition2'] == 'RRNe'), 1, 0)
        X_['_Condition_RRAe'] = np.where((df['Condition1'] == 'RRAe') | (df['Condition2'] == 'RRAe'), 1, 0)
        if self.remove_original:
            X_ = X_.drop(['Condition1'], axis=1)
            X_ = X_.drop(['Condition2'], axis=1)

        ### "BldgType" - default value np.nan
        X_['_BldgType'] = np.where(df['BldgType'].isnull(), np.nan, df['BldgType'].values)
        if self.remove_original:
            X_ = X_.drop(['BldgType'], axis=1)

        ### "HouseStyle" - default value np.nan
        X_['_HouseStyle'] = np.where(df['HouseStyle'].isnull(), np.nan, df['HouseStyle'].values)
        if self.remove_original:
            X_ = X_.drop(['HouseStyle'], axis=1)

        ### "OverallQual" - default value np.nan
        X_['_OverallQual'] = np.where(df['OverallQual'].isnull(), np.nan, df['OverallQual'].values)
        if self.remove_original:
            X_ = X_.drop(['OverallQual'], axis=1)

        ### "OverallCond" - default value np.nan
        X_['_OverallCond'] = np.where(df['OverallCond'].isnull(), np.nan, df['OverallCond'].values)
        if self.remove_original:
            X_ = X_.drop(['OverallCond'], axis=1)

        ### "YearBuilt YearRemodAdd" - default value np.nan
        X_['_YearBuilt'] = np.where(df['YearBuilt'].isnull(), np.nan, df['YearBuilt'].values)
        X_['_YearRemodAdd'] = np.where(df['YearRemodAdd'].isnull(), np.nan, df['YearRemodAdd'].values)
        X_['_BuildingAge'] = df['YrSold'] - df['YearBuilt']
        X_['_YearsFromRemod'] = df['YrSold'] - df['YearRemodAdd']
        if self.remove_original:
            X_ = X_.drop(['YearBuilt'], axis=1)
            X_ = X_.drop(['YearRemodAdd'], axis=1)

        ### "RoofStyle" - default value np.nan
        X_['_RoofStyle'] = np.where(df['RoofStyle'].isnull(), np.nan, df['RoofStyle'].values)
        if self.remove_original:
            X_ = X_.drop(['RoofStyle'], axis=1)

        ### "RoofMatl" - default value np.nan
        X_['_RoofMatl'] = np.where(df['RoofMatl'].isnull(), np.nan, df['RoofMatl'].values)
        if self.remove_original:
            X_ = X_.drop(['RoofMatl'], axis=1)

        ### Combine Exterior1st and Exterior2nd - default value 0
        X_['_Exterior_AsbShng'] = np.where((df['Exterior1st'] == 'AsbShng') | (df['Exterior2nd'] == 'AsbShng'), 1, 0)
        X_['_Exterior_AsphShn'] = np.where((df['Exterior1st'] == 'AsphShn') | (df['Exterior2nd'] == 'AsphShn'), 1, 0)
        X_['_Exterior_BrkComm'] = np.where((df['Exterior1st'] == 'BrkComm') | (df['Exterior2nd'] == 'BrkComm'), 1, 0)
        X_['_Exterior_BrkFace'] = np.where((df['Exterior1st'] == 'BrkFace') | (df['Exterior2nd'] == 'BrkFace'), 1, 0)
        X_['_Exterior_CBlock'] = np.where((df['Exterior1st'] == 'CBlock') | (df['Exterior2nd'] == 'CBlock'), 1, 0)
        X_['_Exterior_CemntBd'] = np.where((df['Exterior1st'] == 'CemntBd') | (df['Exterior2nd'] == 'CemntBd'), 1, 0)
        X_['_Exterior_HdBoard'] = np.where((df['Exterior1st'] == 'HdBoard') | (df['Exterior2nd'] == 'HdBoard'), 1, 0)
        X_['_Exterior_ImStucc'] = np.where((df['Exterior1st'] == 'ImStucc') | (df['Exterior2nd'] == 'ImStucc'), 1, 0)
        X_['_Exterior_MetalSd'] = np.where((df['Exterior1st'] == 'MetalSd') | (df['Exterior2nd'] == 'MetalSd'), 1, 0)
        X_['_Exterior_Plywood'] = np.where((df['Exterior1st'] == 'Plywood') | (df['Exterior2nd'] == 'Plywood'), 1, 0)
        X_['_Exterior_Other'] = np.where((df['Exterior1st'] == 'Other') | (df['Exterior2nd'] == 'Other'), 1, 0)
        X_['_Exterior_Stone'] = np.where((df['Exterior1st'] == 'Stone') | (df['Exterior2nd'] == 'Stone'), 1, 0)
        X_['_Exterior_Stucco'] = np.where((df['Exterior1st'] == 'Stucco') | (df['Exterior2nd'] == 'Stucco'), 1, 0)
        X_['_Exterior_VinylSd'] = np.where((df['Exterior1st'] == 'VinylSd') | (df['Exterior2nd'] == 'VinylSd'), 1, 0)
        X_['_Exterior_WdSdng'] = np.where((df['Exterior1st'] == 'Wd Sdng') | (df['Exterior2nd'] == 'Wd Sdng'), 1, 0)
        X_['_Exterior_WdShing'] = np.where((df['Exterior1st'] == 'WdShing') | (df['Exterior2nd'] == 'WdShing'), 1, 0)
        if self.remove_original:
            X_ = X_.drop(['Exterior1st'], axis=1)
            X_ = X_.drop(['Exterior2nd'], axis=1)

        ### "MasVnrType" - default value np.nan
        X_['_MasVnrType_BrkCmn'] = np.where(df['MasVnrType'] == 'BrkCmn', 1, 0)
        X_['_MasVnrType_BrkFace'] = np.where(df['MasVnrType'] == 'BrkFace', 1, 0)
        X_['_MasVnrType_CBlock'] = np.where(df['MasVnrType'] == 'CBlock', 1, 0)
        X_['_MasVnrType_Stone'] = np.where(df['MasVnrType'] == 'Stone', 1, 0)
        if self.remove_original:
            X_ = X_.drop(['MasVnrType'], axis=1)

        ### "MasVnrArea" - default value np.nan
        X_['_MasVnrArea'] = np.where(df['MasVnrArea'].isnull(), np.nan, df['MasVnrArea'].values)
        if self.remove_original:
            X_ = X_.drop(['MasVnrArea'], axis=1)

        ### "ExterQual" - default value np.nan
        X_['_ExterQual'] = np.nan
        X_['_ExterQual'] = np.where(df['ExterQual'] == 'Ex', 4, X_['_ExterQual'].values)
        X_['_ExterQual'] = np.where(df['ExterQual'] == 'Gd', 3, X_['_ExterQual'].values)
        X_['_ExterQual'] = np.where(df['ExterQual'] == 'TA', 2, X_['_ExterQual'].values)
        X_['_ExterQual'] = np.where(df['ExterQual'] == 'Fa', 1, X_['_ExterQual'].values)
        X_['_ExterQual'] = np.where(df['ExterQual'] == 'Po', 0, X_['_ExterQual'].values)
        if self.remove_original:
            X_ = X_.drop(['ExterQual'], axis=1)

        ### "ExterCond" - default value np.nan
        X_['_ExterCond'] = np.nan
        X_['_ExterCond'] = np.where(df['ExterCond'] == 'Ex', 4, X_['_ExterCond'].values)
        X_['_ExterCond'] = np.where(df['ExterCond'] == 'Gd', 3, X_['_ExterCond'].values)
        X_['_ExterCond'] = np.where(df['ExterCond'] == 'TA', 2, X_['_ExterCond'].values)
        X_['_ExterCond'] = np.where(df['ExterCond'] == 'Fa', 1, X_['_ExterCond'].values)
        X_['_ExterCond'] = np.where(df['ExterCond'] == 'Po', 0, X_['_ExterCond'].values)
        if self.remove_original:
            X_ = X_.drop(['ExterCond'], axis=1)

        ### "Foundation" - default value np.nan
        X_['_Foundation'] = np.where(df['Foundation'].isnull(), np.nan, df['Foundation'].values)
        if self.remove_original:
            X_ = X_.drop(['Foundation'], axis=1)

        ### "BsmtQual" - default value 0
        X_['_BsmtQual'] = 0
        X_['_BsmtQual'] = np.where(df['BsmtQual'] == 'Ex', 5, X_['_BsmtQual'].values)
        X_['_BsmtQual'] = np.where(df['BsmtQual'] == 'Gd', 4, X_['_BsmtQual'].values)
        X_['_BsmtQual'] = np.where(df['BsmtQual'] == 'TA', 3, X_['_BsmtQual'].values)
        X_['_BsmtQual'] = np.where(df['BsmtQual'] == 'Fa', 2, X_['_BsmtQual'].values)
        X_['_BsmtQual'] = np.where(df['BsmtQual'] == 'Po', 1, X_['_BsmtQual'].values)
        if self.remove_original:
            X_ = X_.drop(['BsmtQual'], axis=1)


        return X_


if __name__ == '__main__':
    verbose = False

    # Using modin
    """
    try:
        ray.init()
        ray.init(address='auto', _redis_password='5241590000000000')
    except:
        ray.shutdown()
        ray.init()
    """

    # make all dataframe columns visible
    pd.set_option('display.max_columns', None)

    df = pd.read_csv('/home/peterpirog/PycharmProjects/BostonHousesTune/preprocessing/train.csv')
    # df.to_excel('/home/peterpirog/PycharmProjects/BostonHousesTune/preprocessing/train.xlsx',
    #            sheet_name='X_train_data',
    #            index=False)
    # print(df.head())

    dkt = DomainKnowledgeTransformer()
    df_out = dkt.fit_transform(X=df)
    df_out.to_excel('/home/peterpirog/PycharmProjects/BostonHousesTune/preprocessing/output_data.xlsx',
                    sheet_name='output_data',
                    index=False)

    print(df_out.head(10))
    print(df_out.info())
    print(df_out.describe())
