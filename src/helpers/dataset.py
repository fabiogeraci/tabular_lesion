import pandas as pd


class DataSet:
    def __init__(self, csv_file_name: str):
        self.imported_dataframe = pd.read_csv(csv_file_name)
        self.training_df: pd.DataFrame() = None
        self.X_train: pd.DataFrame() = None
        self.y_train: pd.DataFrame() = None
        self.X_test: pd.DataFrame() = None
        self.y_test: pd.DataFrame() = None
        self.target_name = ''

        self.generate_train_set()
        self.generate_test_set()
        self.dup_col = None

    def generate_train_set(self):
        """
        Generate the training set
        """
        df_train = self.imported_dataframe[self.imported_dataframe['Inf_Train_test'].str.contains('train', case=False)]
        df_valid = self.imported_dataframe[self.imported_dataframe['Inf_Train_test'].str.contains('valid', case=False)]

        self.training_df = pd.concat([df_train, df_valid], axis=0)

        print(f'Are there any Nan = {self.training_df.isnull().values.any()}, Number of Nan = {self.training_df.isnull().sum().sum()}')

        for key in self.training_df.keys():
            if 'target' in key.lower():
                self.target_name = key
                print(self.target_name)

        self.X_train = self.training_df.drop(['Target_Lesion_ClinSig', 'Inf_Train_test'], axis=1)
        self.X_train = self.drop_all_zero_columns(self.X_train)
        self.X_train = self.drop_columns_std_larger(self.X_train)

        self.dup_col = self.get_duplicate_columns(self.X_train)
        self.X_train = self.X_train.drop(self.dup_col, axis=1)

        self.y_train = self.training_df['Target_Lesion_ClinSig']

    def generate_test_set(self):
        """
        Generate the test set
        """
        df_test = self.imported_dataframe[self.imported_dataframe['Inf_Train_test'].str.contains('test', case=False)]
        self.X_test = df_test.drop(['Target_Lesion_ClinSig', 'Inf_Train_test'], axis=1)
        self.X_test = self.drop_all_zero_columns(self.X_test)
        self.X_test = self.drop_columns_std_larger(self.X_test)

        self.X_test = self.X_test.drop(self.dup_col, axis=1)

        self.y_test = df_test['Target_Lesion_ClinSig']

    @staticmethod
    def drop_all_zero_columns(a_dataframe: pd.DataFrame) -> pd.DataFrame:
        """

        :param a_dataframe:
        :return:
        """
        return a_dataframe.loc[:, a_dataframe.ne(0).any()]

    @staticmethod
    def drop_columns_std_larger(a_dataframe: pd.DataFrame) -> pd.DataFrame:
        """

        :param a_dataframe:
        :return:
        """
        return a_dataframe.loc[:, a_dataframe.std() < 10000]

    @staticmethod
    def get_duplicate_columns(df: pd.DataFrame) -> list:

        # Create an empty set
        duplicate_column_names = set()

        # Iterate through all the columns
        # of dataframe
        for x in range(df.shape[1]):

            # Take column at xth index.
            col = df.iloc[:, x]

            # Iterate through all the columns in
            # DataFrame from (x + 1)th index to
            # last index
            for y in range(x + 1, df.shape[1]):

                # Take column at yth index.
                other_col = df.iloc[:, y]

                # Check if two columns at x & y
                # index are equal or not,
                # if equal then adding
                # to the set
                if col.equals(other_col):
                    duplicate_column_names.add(df.columns.values[y])

        # Return list of unique column names
        # whose contents are duplicates.
        return list(duplicate_column_names)
