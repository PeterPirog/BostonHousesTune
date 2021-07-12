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
        X_['_MSSubClass'] = X_.MSSubClass.map(str)+'_'  #convert to string from integer
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

        ### "BsmtCond" - default value 0
        X_['_BsmtCond'] = 0
        X_['_BsmtCond'] = np.where(df['BsmtCond'] == 'Ex', 5, X_['_BsmtCond'].values)
        X_['_BsmtCond'] = np.where(df['BsmtCond'] == 'Gd', 4, X_['_BsmtCond'].values)
        X_['_BsmtCond'] = np.where(df['BsmtCond'] == 'TA', 3, X_['_BsmtCond'].values)
        X_['_BsmtCond'] = np.where(df['BsmtCond'] == 'Fa', 2, X_['_BsmtCond'].values)
        X_['_BsmtCond'] = np.where(df['BsmtCond'] == 'Po', 1, X_['_BsmtCond'].values)
        if self.remove_original:
            X_ = X_.drop(['BsmtCond'], axis=1)

        ### "BsmtExposure" - default value 0
        X_['_BsmtExposure'] = 0
        X_['_BsmtExposure'] = np.where(df['BsmtExposure'] == 'Gd', 4, X_['_BsmtExposure'].values)
        X_['_BsmtExposure'] = np.where(df['BsmtExposure'] == 'Av', 3, X_['_BsmtExposure'].values)
        X_['_BsmtExposure'] = np.where(df['BsmtExposure'] == 'Mn', 2, X_['_BsmtExposure'].values)
        X_['_BsmtExposure'] = np.where(df['BsmtExposure'] == 'No', 1, X_['_BsmtExposure'].values)
        if self.remove_original:
            X_ = X_.drop(['BsmtExposure'], axis=1)

        ### "BsmtFinType1" - default value 0
        X_['_BsmtFinType1'] = 0
        X_['_BsmtFinType1'] = np.where(df['BsmtFinType1'] == 'GLQ', 6, X_['_BsmtFinType1'].values)
        X_['_BsmtFinType1'] = np.where(df['BsmtFinType1'] == 'ALQ', 5, X_['_BsmtFinType1'].values)
        X_['_BsmtFinType1'] = np.where(df['BsmtFinType1'] == 'BLQ', 4, X_['_BsmtFinType1'].values)
        X_['_BsmtFinType1'] = np.where(df['BsmtFinType1'] == 'Rec', 3, X_['_BsmtFinType1'].values)
        X_['_BsmtFinType1'] = np.where(df['BsmtFinType1'] == 'LwQ', 2, X_['_BsmtFinType1'].values)
        X_['_BsmtFinType1'] = np.where(df['BsmtFinType1'] == 'Unf', 1, X_['_BsmtFinType1'].values)
        if self.remove_original:
            X_ = X_.drop(['BsmtFinType1'], axis=1)

        ### "BsmtFinSF1" - default value np.nan
        X_['_BsmtFinSF1'] = np.where(df['BsmtFinSF1'].isnull(), np.nan, df['BsmtFinSF1'].values)
        if self.remove_original:
            X_ = X_.drop(['BsmtFinSF1'], axis=1)

        ### "BsmtFinType2" - default value 0
        X_['_BsmtFinType2'] = 0
        X_['_BsmtFinType2'] = np.where(df['BsmtFinType2'] == 'GLQ', 6, X_['_BsmtFinType2'].values)
        X_['_BsmtFinType2'] = np.where(df['BsmtFinType2'] == 'ALQ', 5, X_['_BsmtFinType2'].values)
        X_['_BsmtFinType2'] = np.where(df['BsmtFinType2'] == 'BLQ', 4, X_['_BsmtFinType2'].values)
        X_['_BsmtFinType2'] = np.where(df['BsmtFinType2'] == 'Rec', 3, X_['_BsmtFinType2'].values)
        X_['_BsmtFinType2'] = np.where(df['BsmtFinType2'] == 'LwQ', 2, X_['_BsmtFinType2'].values)
        X_['_BsmtFinType2'] = np.where(df['BsmtFinType2'] == 'Unf', 1, X_['_BsmtFinType2'].values)
        if self.remove_original:
            X_ = X_.drop(['BsmtFinType2'], axis=1)

        ### "BsmtFinSF2" - default value np.nan
        X_['_BsmtFinSF2'] = np.where(df['BsmtFinSF2'].isnull(), np.nan, df['BsmtFinSF2'].values)
        if self.remove_original:
            X_ = X_.drop(['BsmtFinSF2'], axis=1)

        ### "BsmtUnfSF" - default value np.nan
        X_['_BsmtUnfSF'] = np.where(df['BsmtUnfSF'].isnull(), np.nan, df['BsmtUnfSF'].values)
        if self.remove_original:
            X_ = X_.drop(['BsmtUnfSF'], axis=1)

        ### "TotalBsmtSF" - default value np.nan
        X_['_TotalBsmtSF'] = np.where(df['TotalBsmtSF'].isnull(), np.nan, df['TotalBsmtSF'].values)
        if self.remove_original:
            X_ = X_.drop(['TotalBsmtSF'], axis=1)

        ### "Heating" - default value np.nan
        X_['_Heating'] = np.where(df['Heating'].isnull(), np.nan, df['Heating'].values)
        if self.remove_original:
            X_ = X_.drop(['Heating'], axis=1)

        ### "HeatingQC" - default value np.nan
        X_['_HeatingQC'] = np.nan
        X_['_HeatingQC'] = np.where(df['HeatingQC'] == 'Ex', 4, X_['_HeatingQC'].values)
        X_['_HeatingQC'] = np.where(df['HeatingQC'] == 'Gd', 3, X_['_HeatingQC'].values)
        X_['_HeatingQC'] = np.where(df['HeatingQC'] == 'TA', 2, X_['_HeatingQC'].values)
        X_['_HeatingQC'] = np.where(df['HeatingQC'] == 'Fa', 1, X_['_HeatingQC'].values)
        X_['_HeatingQC'] = np.where(df['HeatingQC'] == 'Po', 0, X_['_HeatingQC'].values)
        if self.remove_original:
            X_ = X_.drop(['HeatingQC'], axis=1)

        ### "CentralAir" - default value 0
        X_['_CentralAir'] = np.where(df['CentralAir'] == 'Y', 1, 0)
        if self.remove_original:
            X_ = X_.drop(['CentralAir'], axis=1)

        ### "Electrical" - default value np.nan
        X_['_Electrical'] = np.nan
        X_['_Electrical'] = np.where(df['Electrical'] == 'SBrkr', 4, X_['_Electrical'].values)
        X_['_Electrical'] = np.where(df['Electrical'] == 'FuseA', 3, X_['_Electrical'].values)
        X_['_Electrical'] = np.where(df['Electrical'] == 'FuseF', 2, X_['_Electrical'].values)
        X_['_Electrical'] = np.where(df['Electrical'] == 'FuseP', 1, X_['_Electrical'].values)
        X_['_Electrical'] = np.where(df['Electrical'] == 'Mix', 0, X_['_Electrical'].values)
        if self.remove_original:
            X_ = X_.drop(['Electrical'], axis=1)

        ### "1stFlrSF" - default value np.nan
        X_['_1stFlrSF'] = np.where(df['1stFlrSF'].isnull(), np.nan, df['1stFlrSF'].values)
        if self.remove_original:
            X_ = X_.drop(['1stFlrSF'], axis=1)

        ### "2ndFlrSF" - default value np.nan
        X_['_2ndFlrSF'] = np.where(df['2ndFlrSF'].isnull(), np.nan, df['2ndFlrSF'].values)
        if self.remove_original:
            X_ = X_.drop(['2ndFlrSF'], axis=1)

        ### "LowQualFinSF" - default value np.nan
        X_['_LowQualFinSF'] = np.where(df['LowQualFinSF'].isnull(), np.nan, df['LowQualFinSF'].values)
        if self.remove_original:
            X_ = X_.drop(['LowQualFinSF'], axis=1)

        ### "GrLivArea" - default value np.nan
        X_['_GrLivArea'] = np.where(df['GrLivArea'].isnull(), np.nan, df['GrLivArea'].values)
        if self.remove_original:
            X_ = X_.drop(['GrLivArea'], axis=1)

        ### "BsmtFullBath" - default value np.nan
        X_['_BsmtFullBath'] = np.where(df['BsmtFullBath'].isnull(), np.nan, df['BsmtFullBath'].values)
        if self.remove_original:
            X_ = X_.drop(['BsmtFullBath'], axis=1)

        ### "BsmtHalfBath" - default value np.nan
        X_['_BsmtHalfBath'] = np.where(df['BsmtHalfBath'].isnull(), np.nan, df['BsmtHalfBath'].values)
        if self.remove_original:
            X_ = X_.drop(['BsmtHalfBath'], axis=1)

        ### "FullBath" - default value np.nan
        X_['_FullBath'] = np.where(df['FullBath'].isnull(), np.nan, df['FullBath'].values)
        if self.remove_original:
            X_ = X_.drop(['FullBath'], axis=1)

        ### "HalfBath" - default value np.nan
        X_['_HalfBath'] = np.where(df['HalfBath'].isnull(), np.nan, df['HalfBath'].values)
        if self.remove_original:
            X_ = X_.drop(['HalfBath'], axis=1)

        ### "BedroomAbvGr" - default value np.nan
        X_['_BedroomAbvGr'] = np.where(df['BedroomAbvGr'].isnull(), np.nan, df['BedroomAbvGr'].values)
        if self.remove_original:
            X_ = X_.drop(['BedroomAbvGr'], axis=1)

        ### "KitchenAbvGr" - default value np.nan
        X_['_KitchenAbvGr'] = np.where(df['KitchenAbvGr'].isnull(), np.nan, df['KitchenAbvGr'].values)
        if self.remove_original:
            X_ = X_.drop(['KitchenAbvGr'], axis=1)

        ### "KitchenQual" - default value np.nan
        X_['_KitchenQual'] = np.nan
        X_['_KitchenQual'] = np.where(df['KitchenQual'] == 'Ex', 4, X_['_KitchenQual'].values)
        X_['_KitchenQual'] = np.where(df['KitchenQual'] == 'Gd', 3, X_['_KitchenQual'].values)
        X_['_KitchenQual'] = np.where(df['KitchenQual'] == 'TA', 2, X_['_KitchenQual'].values)
        X_['_KitchenQual'] = np.where(df['KitchenQual'] == 'Fa', 1, X_['_KitchenQual'].values)
        X_['_KitchenQual'] = np.where(df['KitchenQual'] == 'Po', 0, X_['_KitchenQual'].values)
        if self.remove_original:
            X_ = X_.drop(['KitchenQual'], axis=1)

        ### "TotRmsAbvGrd" - default value np.nan
        X_['_TotRmsAbvGrd'] = np.where(df['TotRmsAbvGrd'].isnull(), np.nan, df['TotRmsAbvGrd'].values)
        if self.remove_original:
            X_ = X_.drop(['TotRmsAbvGrd'], axis=1)

        ### "Functional" - default value np.nan
        X_['_Functional'] = np.nan
        X_['_Functional'] = np.where(df['Functional'] == 'Typ', 7, X_['_Functional'].values)
        X_['_Functional'] = np.where(df['Functional'] == 'Min1', 6, X_['_Functional'].values)
        X_['_Functional'] = np.where(df['Functional'] == 'Min2', 5, X_['_Functional'].values)
        X_['_Functional'] = np.where(df['Functional'] == 'Mod', 4, X_['_Functional'].values)
        X_['_Functional'] = np.where(df['Functional'] == 'Maj1', 3, X_['_Functional'].values)
        X_['_Functional'] = np.where(df['Functional'] == 'Maj2', 2, X_['_Functional'].values)
        X_['_Functional'] = np.where(df['Functional'] == 'Sev', 1, X_['_Functional'].values)
        X_['_Functional'] = np.where(df['Functional'] == 'Sal', 0, X_['_Functional'].values)
        if self.remove_original:
            X_ = X_.drop(['Functional'], axis=1)

        ### "Fireplaces" - default value np.nan
        X_['_Fireplaces'] = np.where(df['Fireplaces'].isnull(), np.nan, df['Fireplaces'].values)
        if self.remove_original:
            X_ = X_.drop(['Fireplaces'], axis=1)

        ### "FireplaceQu" - default value 0
        X_['_FireplaceQu'] = 0
        X_['_FireplaceQu'] = np.where(df['FireplaceQu'] == 'Ex', 4, X_['_FireplaceQu'].values)
        X_['_FireplaceQu'] = np.where(df['FireplaceQu'] == 'Gd', 3, X_['_FireplaceQu'].values)
        X_['_FireplaceQu'] = np.where(df['FireplaceQu'] == 'TA', 2, X_['_FireplaceQu'].values)
        X_['_FireplaceQu'] = np.where(df['FireplaceQu'] == 'Fa', 1, X_['_FireplaceQu'].values)
        X_['_FireplaceQu'] = np.where(df['FireplaceQu'] == 'Po', 0, X_['_FireplaceQu'].values)
        if self.remove_original:
            X_ = X_.drop(['FireplaceQu'], axis=1)

        ### GarageType - default value 0
        X_['_GarageType_2Types'] = np.where(df['GarageType'] == '2Types', 1, 0)
        X_['_GarageType_Attchd'] = np.where(df['GarageType'] == 'Attchd', 1, 0)
        X_['_GarageType_Basment'] = np.where(df['GarageType'] == 'Basment', 1, 0)
        X_['_GarageType_BuiltIn'] = np.where(df['GarageType'] == 'BuiltIn', 1, 0)
        X_['_GarageType_CarPort'] = np.where(df['GarageType'] == 'CarPort', 1, 0)
        X_['_GarageType_Detchd'] = np.where(df['GarageType'] == 'Detchd', 1, 0)
        if self.remove_original:
            X_ = X_.drop(['GarageType'], axis=1)

        ### "GarageYrBlt" - default value np.nan
        X_['_GarageYrBlt'] = np.where(df['GarageYrBlt'].isnull(), np.nan, df['GarageYrBlt'].values)
        X_['_GarageAge'] = df['YrSold'] - df['GarageYrBlt']
        if self.remove_original:
            X_ = X_.drop(['GarageYrBlt'], axis=1)

        ### "GarageFinish" - default value 0
        X_['_GarageFinish'] = 0
        X_['_GarageFinish'] = np.where(df['GarageFinish'] == 'Fin', 3, X_['_GarageFinish'].values)
        X_['_GarageFinish'] = np.where(df['GarageFinish'] == 'RFn', 2, X_['_GarageFinish'].values)
        X_['_GarageFinish'] = np.where(df['GarageFinish'] == 'Unf', 1, X_['_GarageFinish'].values)
        if self.remove_original:
            X_ = X_.drop(['GarageFinish'], axis=1)

        ### "GarageCars" - default value 0
        X_['_GarageCars'] = np.where(df['GarageCars'].isnull(), 0, df['GarageCars'].values)
        if self.remove_original:
            X_ = X_.drop(['GarageCars'], axis=1)

        ### "GarageArea" - default value 0
        X_['_GarageArea'] = np.where(df['GarageArea'].isnull(), 0, df['GarageArea'].values)
        if self.remove_original:
            X_ = X_.drop(['GarageArea'], axis=1)

        ### "GarageQual" - default value 0
        X_['_GarageQual'] = 0
        X_['_GarageQual'] = np.where(df['GarageQual'] == 'Ex', 5, X_['_GarageQual'].values)
        X_['_GarageQual'] = np.where(df['GarageQual'] == 'Gd', 4, X_['_GarageQual'].values)
        X_['_GarageQual'] = np.where(df['GarageQual'] == 'TA', 3, X_['_GarageQual'].values)
        X_['_GarageQual'] = np.where(df['GarageQual'] == 'Fa', 2, X_['_GarageQual'].values)
        X_['_GarageQual'] = np.where(df['GarageQual'] == 'Po', 1, X_['_GarageQual'].values)
        if self.remove_original:
            X_ = X_.drop(['GarageQual'], axis=1)

        ### "GarageCond" - default value 0
        X_['_GarageCond'] = 0
        X_['_GarageCond'] = np.where(df['GarageCond'] == 'Ex', 5, X_['_GarageCond'].values)
        X_['_GarageCond'] = np.where(df['GarageCond'] == 'Gd', 4, X_['_GarageCond'].values)
        X_['_GarageCond'] = np.where(df['GarageCond'] == 'TA', 3, X_['_GarageCond'].values)
        X_['_GarageCond'] = np.where(df['GarageCond'] == 'Fa', 2, X_['_GarageCond'].values)
        X_['_GarageCond'] = np.where(df['GarageCond'] == 'Po', 1, X_['_GarageCond'].values)
        if self.remove_original:
            X_ = X_.drop(['GarageCond'], axis=1)

        ### "PavedDrive" - default valuenp.nan
        X_['_PavedDrive'] = np.nan
        X_['_PavedDrive'] = np.where(df['PavedDrive'] == 'Y', 2, X_['_PavedDrive'].values)
        X_['_PavedDrive'] = np.where(df['PavedDrive'] == 'P', 1, X_['_PavedDrive'].values)
        X_['_PavedDrive'] = np.where(df['PavedDrive'] == 'N', 0, X_['_PavedDrive'].values)
        if self.remove_original:
            X_ = X_.drop(['PavedDrive'], axis=1)

        ### "WoodDeckSF" - default value 0
        X_['_WoodDeckSF'] = np.where(df['WoodDeckSF'].isnull(), 0, df['WoodDeckSF'].values)
        if self.remove_original:
            X_ = X_.drop(['WoodDeckSF'], axis=1)

        ### "OpenPorchSF" - default value 0
        X_['_OpenPorchSF'] = np.where(df['OpenPorchSF'].isnull(), 0, df['OpenPorchSF'].values)
        if self.remove_original:
            X_ = X_.drop(['OpenPorchSF'], axis=1)

        ### "EnclosedPorch" - default value 0
        X_['_EnclosedPorch'] = np.where(df['EnclosedPorch'].isnull(), 0, df['EnclosedPorch'].values)
        if self.remove_original:
            X_ = X_.drop(['EnclosedPorch'], axis=1)

        ### "3SsnPorch" - default value 0
        X_['_3SsnPorch'] = np.where(df['3SsnPorch'].isnull(), 0, df['3SsnPorch'].values)
        if self.remove_original:
            X_ = X_.drop(['3SsnPorch'], axis=1)

        ### "ScreenPorch" - default value 0
        X_['_ScreenPorch'] = np.where(df['ScreenPorch'].isnull(), 0, df['ScreenPorch'].values)
        if self.remove_original:
            X_ = X_.drop(['ScreenPorch'], axis=1)

        ### "PoolArea" - default value 0
        X_['_PoolArea'] = np.where(df['PoolArea'].isnull(), 0, df['PoolArea'].values)
        if self.remove_original:
            X_ = X_.drop(['PoolArea'], axis=1)

        ### "PoolQC" - default value 0
        X_['_PoolQC'] = 0
        X_['_PoolQC'] = np.where(df['PoolQC'] == 'Ex', 4, X_['_PoolQC'].values)
        X_['_PoolQC'] = np.where(df['PoolQC'] == 'Gd', 3, X_['_PoolQC'].values)
        X_['_PoolQC'] = np.where(df['PoolQC'] == 'TA', 2, X_['_PoolQC'].values)
        X_['_PoolQC'] = np.where(df['PoolQC'] == 'Fa', 1, X_['_PoolQC'].values)
        if self.remove_original:
            X_ = X_.drop(['PoolQC'], axis=1)

        ### "Fence" - default value 0
        X_['_Fence'] = 0
        X_['_Fence'] = np.where(df['Fence'] == 'GdPrv', 4, X_['_Fence'].values)
        X_['_Fence'] = np.where(df['Fence'] == 'MnPrv', 3, X_['_Fence'].values)
        X_['_Fence'] = np.where(df['Fence'] == 'GdWo', 2, X_['_Fence'].values)
        X_['_Fence'] = np.where(df['Fence'] == 'MnWo', 1, X_['_Fence'].values)
        if self.remove_original:
            X_ = X_.drop(['Fence'], axis=1)

        ### MiscFeature - default value 0
        X_['_MiscFeature_Elev'] = np.where(df['MiscFeature'] == 'Elev', 1, 0)
        X_['_MiscFeature_Gar2'] = np.where(df['MiscFeature'] == 'Gar2', 1, 0)
        X_['_MiscFeature_Shed'] = np.where(df['MiscFeature'] == 'Shed', 1, 0)
        X_['_MiscFeature_TenC'] = np.where(df['MiscFeature'] == 'TenC', 1, 0)
        if self.remove_original:
            X_ = X_.drop(['MiscFeature'], axis=1)

        ### "MiscVal" - default value 0
        X_['_MiscVal'] = np.where(df['MiscVal'].isnull(), 0, df['MiscVal'].values)
        if self.remove_original:
            X_ = X_.drop(['MiscVal'], axis=1)

        ### "MoSold" - default value np.nan
        X_['_MoSold'] = np.where(df['MoSold'].isnull(), np.nan, df['MoSold'].values)
        X_['_QuarterSold']=1+(X_['_MoSold']-1)//3
        if self.remove_original:
            X_ = X_.drop(['MoSold'], axis=1)

        ### "YrSold" - default value np.nan
        X_['_YrSold'] = np.where(df['YrSold'].isnull(), np.nan, df['YrSold'].values)
        if self.remove_original:
            X_ = X_.drop(['YrSold'], axis=1)


        ### "SaleType" - default value np.nan
        X_['_SaleType_WD'] = np.where(df['SaleType'] == 'WD', 1, 0)
        X_['_SaleType_CWD'] = np.where(df['SaleType'] == 'CWD', 1, 0)
        X_['_SaleType_VWD'] = np.where(df['SaleType'] == 'VWD', 1, 0)
        X_['_SaleType_New'] = np.where(df['SaleType'] == 'New', 1, 0)
        X_['_SaleType_COD'] = np.where(df['SaleType'] == 'COD', 1, 0)
        X_['_SaleType_Con'] = np.where(df['SaleType'] == 'Con', 1, 0)
        X_['_SaleType_ConLw'] = np.where(df['SaleType'] == 'ConLw', 1, 0)
        X_['_SaleType_ConLI'] = np.where(df['SaleType'] == 'ConLI', 1, 0)
        X_['_SaleType_ConLD'] = np.where(df['SaleType'] == 'ConLD', 1, 0)
        if self.remove_original:
            X_ = X_.drop(['SaleType'], axis=1)

        ### "SaleCondition" - default value np.nan
        X_['_SaleCondition_Normal'] = np.where(df['SaleCondition'] == 'Normal', 1, 0)
        X_['_SaleCondition_Abnorml'] = np.where(df['SaleCondition'] == 'Abnorml', 1, 0)
        X_['_SaleCondition_AdjLand'] = np.where(df['SaleCondition'] == 'AdjLand', 1, 0)
        X_['_SaleCondition_Alloca'] = np.where(df['SaleCondition'] == 'Alloca', 1, 0)
        X_['_SaleCondition_Family'] = np.where(df['SaleCondition'] == 'Family', 1, 0)
        X_['_SaleCondition_Partial'] = np.where(df['SaleCondition'] == 'Partial', 1, 0)
        if self.remove_original:
            X_ = X_.drop(['SaleCondition'], axis=1)

        return X_


if __name__ == '__main__':
    verbose = False

    # Using modin
    """
    try:
        ray.init()
        #ray.init(address='auto', _redis_password='5241590000000000')
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
    import joblib
    from ray.util.joblib import register_ray

    register_ray()
    with joblib.parallel_backend('ray'):
        df_out = dkt.fit_transform(X=df)
        df_out.to_csv(path_or_buf='/home/peterpirog/PycharmProjects/BostonHousesTune/preprocessing/preprocessed_train_data.csv',
                      sep=',',
                      header=True,
                      index=False)
        df_out.to_excel('/home/peterpirog/PycharmProjects/BostonHousesTune/preprocessing/preprocessed_train_data.xlsx',
                        sheet_name='output_data',
                        index=False)

        print(df_out.head(10))
        print(df_out.info())
        print(df_out.describe())
