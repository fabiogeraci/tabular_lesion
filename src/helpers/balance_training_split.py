import pandas as pd
import os
import copy


class TrainSplitDataConfig:
    PATH = os.getcwd()
    PERCENTAGE_TEST_SPLIT = 0.9
    PERCENTAGE_TRAIN_VALID_SPLIT = 0.8
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'
    TRAIN_VALID_TEST = [TRAIN, VALID, TEST]


class TrainingInput:
    target_list = []
    features = []
    features_df = pd.DataFrame()

    def __init__(self, path: str, file):

        self.path = path
        self.file_name = file.lower()
        for file_name in os.listdir(self.path):
            self.add_features(os.path.join(self.path, file_name))
        self.features_df = pd.concat(self.features, axis=0)
        self.set_target_list()

    def add_features(self, file_path):
        if not os.path.isfile(file_path) or '.csv' not in file_path.lower():
            return

        if self.file_name in file_path.lower():
            df = pd.read_csv(file_path, index_col=False)
            self.features.append(df)

    def set_target_list(self):
        """
        Method to set the target list
        """
        for col_name in self.features_df.keys():
            if 'target' in col_name.lower():
                self.target_list.append(col_name)

    @staticmethod
    def split_dataframe(a_dataframe: pd.DataFrame, target: str, split_percentage: float):
        """
        Method to balance each split by target
        :param a_dataframe:
        :param target: target
        :param split_percentage: split percentage
        :return: two dataframes
        """
        groups = a_dataframe.groupby(target)
        row_nums = groups.cumcount()
        sizes = groups[target].transform('size')

        temp = a_dataframe[row_nums <= sizes * split_percentage]
        test = a_dataframe[row_nums > sizes * split_percentage]
        return temp, test

    @staticmethod
    def train_valid_test(a_df: pd.DataFrame, train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, target: str):
        """
        This method overwrites the Inf_Train_test column, and then it replaces _set
        :param a_df:
        :param train: train dataframe
        :param valid: valid dataframe
        :param test: test dataframe
        :param target: target
        :return: balanced dataframe
        """
        a_df.loc[a_df.Inf_Patient.isin(train.Inf_Patient) &
                 a_df.Target_Patient_ClinSig.isin(
                     train[target]), 'Inf_Train_test'] = 'train_set'

        a_df.loc[a_df.Inf_Patient.isin(valid.Inf_Patient) &
                 a_df.Target_Patient_ClinSig.isin(
                     valid[target]), 'Inf_Train_test'] = 'valid_set'

        a_df.loc[a_df.Inf_Patient.isin(test.Inf_Patient) &
                 a_df.Target_Patient_ClinSig.isin(
                     test[target]), 'Inf_Train_test'] = 'test_set'

        a_df.Inf_Train_test = a_df.Inf_Train_test.str.replace('_set', '')

        return a_df

    @staticmethod
    def percentage_split_print(a_dict: dict, a_set_list: list, print_flag: bool = False):
        """
        Calculates the percentage of each split against the total number of samples
        :param a_set_list:
        :param a_dict: T2_axial resampled original image
        :param a_set_list: Train/Valid/Test set name
        :param print_flag: Enables printing Default is False
        :return:
        """

        for target in a_dict.keys():
            target_df = a_dict[target]
            for key in sorted(target_df[target].value_counts().keys()):

                for split_set in a_set_list:

                    pc_pos = target_df.loc[target_df.Inf_Train_test == split_set, target].value_counts()[key] \
                             / (target_df.loc[target_df.Inf_Train_test == split_set, target].value_counts().sum())

                    if print_flag:
                        print(f'{target}_{int(key)}, pc_slit_{split_set} = {pc_pos:.2%}')


class TrainingSplitter:

    @staticmethod
    def write_dataframe_to_csv(split_data: dict, training_input: TrainingInput, path: str):
        """
        This method write out the dataframe
        :param split_data:
        :param training_input: Main Class
        :param path: Store path
        """
        for target in training_input.target_list:
            csv_file_name = f'{training_input.file_name[:-1]}_balanced_{target}.csv'
            split_data[target].to_csv(os.path.join(path, csv_file_name), index=False)

    @staticmethod
    def clean_dataframe(split_data: dict) -> dict:
        """

        :param split_data:
        :return:
        """
        dictionary_copied = copy.deepcopy(split_data)

        for target_key in dictionary_copied.keys():
            for key in dictionary_copied[target_key]:
                if 'Inf_Train_Test'.lower() not in key.lower() and 'Target_'.lower() not in key.lower() and 'Feature_'.lower() not in key.lower():
                    del dictionary_copied[target_key][key]
                elif key in ['Unnamed: 0', 'Inf_Dataset', 'Inf_Patient', 'Inf_Study_Date', 'Inf_Lesion']:
                    del dictionary_copied[target_key][key]

        return dictionary_copied

    @staticmethod
    def rename_features(a_dict: dict) -> dict:
        """

        :param a_dict:
        :return:
        """
        for key in a_dict.keys():
            for idx, features_key in enumerate(a_dict[key].keys()):
                if 'feature_' in features_key.lower():
                    a_dict[key][f'Feature_{idx}'] = a_dict[key].pop(features_key)

        return a_dict

    @staticmethod
    def remove_columns(split_data: dict, training_input: TrainingInput) -> dict:
        """
        This method removes the additional targets, therefore each dataframe will contain one target
        :param split_data:
        :param training_input: Main Class
        :return: a single target dataframe
        """
        for target in training_input.target_list:
            drop_columns = copy.deepcopy(training_input.target_list)
            drop_columns.remove(target)
            split_data[target].drop(drop_columns, axis=1, inplace=True)
        return split_data

    @staticmethod
    def get_split_targets(training_input: TrainingInput, path: str, percentage_test: float,
                          percentage_train_valid: float, sets_list: list) -> dict:
        """
        This method makes a balanced dataset by target, and writes it out into a csv
        :param sets_list:
        :param training_input:
        :param path: csv save file
        :param percentage_test: percentage to apply to test set
        :param percentage_train_valid: percentage to apply to train set
        :return: balanced dataframe by target
        """
        split_data = {}
        for target in training_input.target_list:
            copy_of_features_df = training_input.features_df.copy()

            temp, test = TrainingInput.split_dataframe(copy_of_features_df, target, percentage_test)
            train, valid = TrainingInput.split_dataframe(temp, target, percentage_train_valid)

            copy_of_features_df = TrainingInput.train_valid_test(copy_of_features_df, train, valid, test, target)

            split_data[target] = copy_of_features_df

        split_data = TrainingSplitter.remove_columns(split_data, training_input)

        clean_data = TrainingSplitter.clean_dataframe(split_data)

        TrainingInput.percentage_split_print(clean_data, sets_list, False)

        renamed_dictionary = TrainingSplitter.rename_features(clean_data)

        TrainingSplitter.write_dataframe_to_csv(renamed_dictionary, training_input, path)

        return split_data


if __name__ == '__main__':
    lesion_training_input = TrainingInput(TrainSplitDataConfig.PATH, 'lesion_df.')
    lesion_training_splitter = TrainingSplitter.get_split_targets(lesion_training_input, TrainSplitDataConfig.PATH,
                                                                  TrainSplitDataConfig.PERCENTAGE_TEST_SPLIT,
                                                                  TrainSplitDataConfig.PERCENTAGE_TRAIN_VALID_SPLIT,
                                                                  TrainSplitDataConfig.TRAIN_VALID_TEST)
