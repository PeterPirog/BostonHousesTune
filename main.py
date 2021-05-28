import pandas as pd
from collections import OrderedDict


class DatasetKnowledge():
    def __init__(self, csv_file_path):
        super().__init__()

        self.file_path = csv_file_path
        self.data_original = pd.read_csv(self.file_path)
        self.X = self.data_original.copy()
        self.Y = pd.DataFrame()
        self.columns = self.data_original.columns.to_list()
        self.columns_number=len(self.columns)

        # column functions
        self.targets = []
        self.targets_number = 0

        self.features = self.columns
        self.features_number=len(self.features)

        #Numerical features
        self.numerical_features = list(self.X._get_numeric_data().columns)
        self.numerical_features_number=len(self.numerical_features)
        self.numerical_features_circular = []
        self.numerical_features_circular_number=0

        #Categorical features
        self.categorical_features = self.X.select_dtypes(include=['object']).columns.tolist()
        self.categorical_features_number=len(self.categorical_features)
        self.categorical_features_nominal = self.categorical_features #in the first step we assume that data are nominal
        self.categorical_features_nominal_number=len(self.categorical_features_nominal)
        self.categorical_features_ordinal = []
        self.categorical_features_ordinal_dict = {}
        self.categorical_features_ordinal_number=0

        self.cardinality={}
        self.categorical_features_unique_labels={}

        #Other features
        self.features_text = []
        self.features_text_number=len(self.features_text)
        self.features_removed = []
        self.features_removed_number=len(self.features_removed)

    def add_ordinal_category(self,label,ordered_feature_value_list):
        self.__remove_label_from_lists(label)
        self.categorical_features.append(label)
        self.categorical_features_ordinal.append(label)
        self.categorical_features_ordinal_dict[label]=ordered_feature_value_list

        for key in ordered_feature_value_list:
            if key in self.X[label]:
                pass
            else:
                print(f'Warning value: {key} not exist in column named: {label}')

        self.__update_XY()


    def __update_XY(self):
        """
        Function count number of numerical and categorical features
        :return:
        """
        self.X=self.data_original[self.features].copy()
        self.Y=self.data_original[self.targets].copy()
        self.features_number=len(self.features)
        self.targets_number=len(self.targets)

        #Numerical features
        self.numerical_features_number = len(self.numerical_features)
        self.numerical_features_circular_number=len(self.numerical_features_circular)
        #Categorical features
        self.categorical_features_number=len(self.categorical_features)
        self.categorical_features_nominal_number = len(self.categorical_features_nominal)
        self.categorical_features_ordinal_number = len(self.categorical_features_ordinal)
        #Other features
        self.features_text_number=len(self.features_text)
        self.features_removed_number_number=len(self.features_removed)

    def __remove_label_from_lists(self,label):
        try:
            self.features.remove(label)
            self.numerical_features.remove(label)
            self.numerical_features_circular.remove(label)
            self.categorical_features.remove(label)
            self.categorical_features_nominal.remove(label)
            self.categorical_features_ordinal.remove(label)
            if label in self.categorical_features_ordinal_dict:
                del self.categorical_features_ordinal_dict[label]
            self.features_text.remove(label)
        except:
            pass

    def show_info(self):
        print('--------------------------------------')
        print('Dataset summary')
        print(f'Original number of columns in dataset: {self.columns_number}')
        print(f'Features (columns) discarded number: {self.features_removed_number_number}')
        print(f'Targets number: {self.targets_number}')
        print(f'Features number: {self.features_number}')
        print(f'\tNumerical features number: {self.numerical_features_number}')
        print(f'\t\tNumerical features number (circular): {self.numerical_features_circular_number}')
        print(f'\tCategorical features number: {self.categorical_features_number}')
        print(f'\t\t Categorical features number (nominal): {self.categorical_features_nominal_number}')
        print(f'\t\t Categorical features number (ordinal): {self.categorical_features_ordinal_number}')
        print(f'\tText features number: {self.features_text_number}')
        print('-------------------------------------------------')
        print(f'Numerical values: {dataset.numerical_features}')
        print(f'Categorical values: {dataset.categorical_features}')
        self.show_cardinality()


    def define_targets(self, targets_list):
        for target in targets_list:
            self.__remove_label_from_lists(target)
            self.targets.append(target)

        self.__update_XY()

    def remove_features(self,features_list):
        self.features_removed=features_list
        for feature in features_list:
            self.__remove_label_from_lists(feature)
        self.__update_XY()

    def feature_to_categorical(self,features_list):
        pass

    def show_moments(self):
        #Function shows std, skewness and curtosis
        pass
    def show_cardinality(self):
        pass
        self.cardinality=self.X[self.categorical_features].nunique().to_dict()
        self.cardinality = sorted(self.cardinality.items(), key=lambda x: x[1], reverse=True)
        print('\n Categorical features cardinality:')
        for i in self.cardinality:
            key=i[0]
            value=i[1]
            #write existing labels for specific categorical feature
            self.categorical_features_unique_labels[key]=self.X[key].unique()
            print(key, value)


        pass
    def show_missing_values(self):
        pass
    def show_correlation(self):
        pass

# Press the green button in the gutter to run the script.


if __name__ == '__main__':
    dataset = DatasetKnowledge(csv_file_path='train.csv')

    print('\n dataset.features=',dataset.features)
    dataset.define_targets(['SalePrice'])
    print(dataset.features)

    dataset.remove_features(['Id','MSSubClass'])
    print(dataset.targets)
    print(dataset.X.head(10))
    print(dataset.Y)
    dataset.show_info()

    categorical_labels=dataset.categorical_features


