import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
class DataFrameDataset(Dataset):
    def __init__(self, dataframe, input_columns, output_column):
        self.X = torch.tensor(dataframe[input_columns].values, dtype=torch.float32)
        self.y = torch.tensor(dataframe[output_column].values, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def calculate_stats(df, columns, list_columns: list[str] = []):
    """
    Calculate mean and std dev for specified columns.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        columns (list[str]): The columns to calculate stats for.
        list_columns (list[str], optional): The column names that are lists to calculate
            stats for. Defaults to [].

    Returns:
        tuple: A tuple containing the means and stds for the specified columns.
    """
    means, stds = {}, {}

    for col in tqdm(columns):
        if col in list_columns:
            # For list columns, calculate stats for each element
            col_data = np.array([np.fromstring(x.strip('[]'), sep=',') for x in df[col].values if x is not None])
            means[col] = np.mean(col_data)
            stds[col] = np.std(col_data)
        else:
            # For scalar columns, use pandas methods
            means[col] = df[col].mean()
            stds[col] = df[col].std()
    return means, stds

# Function to normalize specified columns in the DataFrame
def normalize_df(df, means, stds, columns):
    df[columns] = (df[columns] - means) / stds
    return df

def unnormalize_mean(pred, target_mean, target_std):
    return pred * target_std + target_mean

def unnormalize_var(var, target_std):
    return var * (target_std ** 2)