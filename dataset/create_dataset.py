import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import datasets
import pyarrow as pa

def check_data_files():
    """Check if required data files exist and return proper paths."""
    import glob
    import os
    
    # Check if the Akebono data file exists
    if not os.path.exists(akebono_file_path):
        raise FileNotFoundError(f"Akebono data file not found at: {akebono_file_path}")
    
    # Check if the OMNI AL/SYM-H directory exists
    omni_dir = os.path.dirname(omni_al_symh_path)
    if not os.path.exists(omni_dir):
        raise FileNotFoundError(f"OMNI AL/SYM-H directory not found at: {omni_dir}")
    
    # Check if the F10.7 directory exists
    f107_dir = os.path.dirname(f107_file_path)
    if not os.path.exists(f107_dir):
        raise FileNotFoundError(f"F10.7 directory not found at: {f107_dir}")
    
    # Check if the KP Index file exists
    if not os.path.exists(kp_file_path):
        raise FileNotFoundError(f"KP Index file not found at: {kp_file_path}")
    
    # Check for .lst files in OMNI AL/SYM-H and F10.7 directories
    al_symh_files = glob.glob(omni_al_symh_path)
    f107_files = glob.glob(f107_file_path)
    
    if not al_symh_files:
        raise FileNotFoundError(f"No .lst files found in: {omni_dir}")
    if not f107_files:
        raise FileNotFoundError(f"No .lst files found in: {f107_dir}")
    
    # Output the file status
    print("Data files found:")
    print(f"Akebono: {akebono_file_path}")
    print(f"OMNI AL/SYM-H files: {len(al_symh_files)} files")
    print(f"F10.7 files: {len(f107_files)} files")
    print(f"KP Index file: {kp_file_path}")
    
    return al_symh_files, f107_files

def print_rows_removed(before_count, after_df, step_description, column_to_check=None):
    after_count = len(after_df)
    rows_removed = before_count - after_count
    print(f"{step_description}:")
    print(f"Rows removed: {rows_removed}")
    print(f"Rows remaining: {after_count}")
    
    if column_to_check:
        min_value = after_df[column_to_check].min()
        max_value = after_df[column_to_check].max()
        print(f"New range of '{column_to_check}': {min_value} to {max_value}")
    
    print("\n")
    return after_count

# Define relative paths to input data
akebono_file_path = os.path.join('input_dataset', 'Akebono_combined.tsv')
kp_file_path = os.path.join('input_dataset', 'omni_kp_index.lst')
omni_al_symh_path = os.path.join('input_dataset', 'omni_al_index_symh', '*.lst')
f107_file_path = os.path.join('input_dataset', 'omni_f107', '*.lst')

al_symh_files, f107_files = check_data_files()

# Read the Akebono data in chunks to optimize memory usage
print("Reading Akebono data...")
df = pd.read_csv(akebono_file_path, sep='\t')
df['DateFormatted'] = pd.to_datetime(df['DateFormatted'], errors='coerce')
initial_row_count = len(df)

print(f"\nInitial number of rows: {initial_row_count}\n")
print("Starting data cleaning steps...")

# Remove rows with '999' values in XXLAT and XXLON
columns_with_999 = ['XXLAT', 'XXLON']
mask_999 = (df[columns_with_999] == 999).any(axis=1)
filtered_df = df[~mask_999]
initial_row_count = print_rows_removed(initial_row_count, filtered_df, "After removing rows with '999' values", column_to_check='GLAT')

# Remove rows with ILAT > 90
filtered_df = filtered_df[filtered_df['ILAT'] <= 90]
initial_row_count = print_rows_removed(initial_row_count, filtered_df, "After removing rows with ILAT > 90", column_to_check='ILAT')

# Filter Altitude between 1000km and 8000km
filtered_df = filtered_df[(filtered_df['Altitude'] >= 1000) & (filtered_df['Altitude'] <= 8000)]
initial_row_count = print_rows_removed(initial_row_count, filtered_df, "After filtering Altitude between 1000km and 8000km", column_to_check='Altitude')

# Convert DateFormatted to datetime, remove NaT values, and filter rows before 1990-01-01
filtered_df['DateFormatted'] = pd.to_datetime(filtered_df['DateFormatted'], errors='coerce')
filtered_df = filtered_df.dropna(subset=['DateFormatted'])
filtered_df = filtered_df[filtered_df['DateFormatted'] >= '1990-01-01']
initial_row_count = print_rows_removed(initial_row_count, filtered_df, "After filtering rows before January 1st, 1990", column_to_check='DateFormatted')

# Combine date and time into a single datetime column
filtered_df['DateTimeFormatted'] = pd.to_datetime(
    filtered_df['DateFormatted'].dt.strftime('%Y-%m-%d') + ' ' + filtered_df['TimeFormatted'].astype(str),
    errors='coerce'
)
filtered_df['DateTimeFormatted'] = filtered_df['DateTimeFormatted'].dt.floor('min')
filtered_df = filtered_df.drop(columns=['DateFormatted', 'TimeFormatted', 'Date', 'Time'], errors='ignore')

for col in ['AL_index', 'SYM_H']:
    if col not in filtered_df.columns:
        filtered_df[col] = np.nan

filtered_df.reset_index(drop=True, inplace=True)
filtered_df.set_index('DateTimeFormatted', inplace=True)

# -----------------------------------
# SYM-H and AL Index Data

# AL_index: Every 10 mins for the prev 5 hours
# SYM_H: Every 30 minutes for the previous 3 days
# -----------------------------------
print("Processing SYM-H and AL index data...")

df_list = []
for file in tqdm(al_symh_files, desc="Loading omni_al_index_symh data"):
    try:
        columns = ['Year', 'Day', 'Hour', 'Minute', 'AL_index', 'SYM_H']
        df_temp = pd.read_csv(file, sep=r'\s+', names=columns)
        df_temp['DateTime'] = pd.to_datetime(df_temp['Year'] * 1000 + df_temp['Day'], format='%Y%j') \
                            + pd.to_timedelta(df_temp['Hour'], unit='h') \
                            + pd.to_timedelta(df_temp['Minute'], unit='m')
        df_temp = df_temp[['DateTime', 'AL_index', 'SYM_H']]
        df_list.append(df_temp)
    except Exception as e:
        print(f"Error reading file {file}: {str(e)}")
        continue

if not df_list:
    raise ValueError("No valid data files were processed. Please ensure your data files exist and are properly formatted.")

omni_df = pd.concat(df_list, ignore_index=True)
del df_list
omni_df.drop_duplicates(subset='DateTime', keep='first', inplace=True)
omni_df.set_index('DateTime', inplace=True)
omni_df.sort_index(inplace=True)
filtered_df.sort_index(inplace=True)

print("Expanding temporal features for SYM-H and AL index...")
al_time_range = pd.timedelta_range(start='0m', end='5h', freq='10min')
sym_h_time_range = pd.timedelta_range(start='0m', end='3d', freq='30min')
dt_index = filtered_df.index
al_timestamps = dt_index.values[:, None] - al_time_range.values
al_timestamps = pd.DatetimeIndex(al_timestamps.ravel())
sym_h_timestamps = dt_index.values[:, None] - sym_h_time_range.values
sym_h_timestamps = pd.DatetimeIndex(sym_h_timestamps.ravel())
al_values = omni_df['AL_index'].reindex(al_timestamps).values.reshape(len(dt_index), -1)
sym_h_values = omni_df['SYM_H'].reindex(sym_h_timestamps).values.reshape(len(dt_index), -1)

print("Creating AL_index temporal features...")
al_columns = {}
for i in tqdm(range(al_values.shape[1]), desc="AL_index columns"):
    al_columns[f'AL_index_{i}'] = al_values[:, i]

print("Creating SYM_H temporal features...")
sym_h_columns = {}
for i in tqdm(range(sym_h_values.shape[1]), desc="SYM_H columns"):
    sym_h_columns[f'SYM_H_{i}'] = sym_h_values[:, i]

print("Concatenating new temporal features to the DataFrame...")
filtered_df = pd.concat([
    filtered_df, 
    pd.DataFrame(al_columns, index=filtered_df.index),
    pd.DataFrame(sym_h_columns, index=filtered_df.index)
], axis=1)
filtered_df.drop(columns=['AL_index', 'SYM_H'], inplace=True, errors='ignore')

# -----------------------------------
# F10.7 Solar Flux Index

# F10.7: Every 24 hours for the prev 72 hours
# -----------------------------------
print("Processing F10.7 solar flux index data...")

f107_list = []
for file in tqdm(f107_files, desc="Loading f107 data"):
    try:
        columns = ['Year', 'Day', 'Hour', 'f107_index']
        f107_df = pd.read_csv(file, sep=r'\s+', names=columns)
        f107_df['DateTime'] = pd.to_datetime(f107_df['Year'] * 1000 + f107_df['Day'], format='%Y%j') \
                              + pd.to_timedelta(f107_df['Hour'], unit='h')
        f107_df = f107_df[['DateTime', 'f107_index']]
        f107_list.append(f107_df)
    except Exception as e:
        print(f"Error reading file {file}: {str(e)}")
        continue

if not f107_list:
    raise ValueError("No valid F10.7 files were processed. Please ensure your data files exist and are properly formatted.")

f107_df_combined = pd.concat(f107_list, ignore_index=True)
del f107_list
f107_df_combined.drop_duplicates(subset='DateTime', keep='first', inplace=True)
f107_df_combined.set_index('DateTime', inplace=True)
f107_df_combined.sort_index(inplace=True)

print("Expanding temporal features for F10.7 index...")
f107_time_range = pd.timedelta_range(start='0h', end='72h', freq='24h')
f107_timestamps = dt_index.values[:, None] - f107_time_range.values
f107_timestamps = pd.DatetimeIndex(f107_timestamps.ravel()).round('h')
f107_values = f107_df_combined['f107_index'].reindex(f107_timestamps).values.reshape(len(dt_index), -1)

print("Creating f107_index temporal features...")
f107_columns = {}
for i in tqdm(range(f107_values.shape[1]), desc="f107_index columns"):
    f107_columns[f'f107_index_{i}'] = f107_values[:, i]

print("Concatenating f107_index temporal features to the DataFrame...")
filtered_df = pd.concat([filtered_df, pd.DataFrame(f107_columns, index=filtered_df.index)], axis=1)

# -----------------------------------
# Kp Index Data
# -----------------------------------
print("Processing Kp index data...")
try:
    kp_df = pd.read_csv(kp_file_path, sep=r'\s+', names=['Year', 'DOY', 'Hour', 'Kp_index'])
    kp_df['DateTime'] = pd.to_datetime(kp_df['Year'] * 1000 + kp_df['DOY'], format='%Y%j') \
                        + pd.to_timedelta(kp_df['Hour'], unit='h')
    kp_df = kp_df[['DateTime', 'Kp_index']]
except Exception as e:
    print(f"Error reading kp index file {kp_file_path}: {str(e)}")
    kp_df = pd.DataFrame(columns=['DateTime', 'Kp_index'])

if kp_df.empty:
    raise ValueError("No valid kp index data was processed. Please ensure your file exists and is properly formatted.")

kp_df.drop_duplicates(subset='DateTime', keep='first', inplace=True)
kp_df.set_index('DateTime', inplace=True)
kp_df.sort_index(inplace=True)

print("Adding nearest hourly Kp index values...")
# Round filtered_df index to nearest hour to match kp_df hourly data
nearest_hours = filtered_df.index.round('h')
filtered_df['Kp_index'] = kp_df['Kp_index'].reindex(nearest_hours).values // 10

# -----------------------------------
# Data Cleaning: Replace Invalid Values and Optimize Data Types
# -----------------------------------
print("Cleaning data and optimizing data types...")

# List of invalid placeholder values
invalid_values = [99, 99.9, 999.9, 9.999, 9999.0, 9999.99, 99999.99, 9999999, 9999999.0]

# Function to count and replace invalid values
def replace_and_count_invalid_values(df, invalid_values, replacement=0):
    """
    Replaces invalid values in the DataFrame and returns a report
    of how many values were updated for each invalid value.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to clean.
        invalid_values (list): A list of invalid values to replace.
        replacement (int/float): The value to replace invalid values with.
    
    Returns:
        dict: A dictionary containing the count of replaced values for each invalid value.
    """
    update_counts = {}
    
    for value in invalid_values:
        # Count how many values match the invalid value
        count = (df == value).sum().sum()
        if count > 0:
            update_counts[value] = count
            # Replace the invalid value with the replacement
            df.replace(value, replacement, inplace=True)
    
    # Generate the report
    if update_counts:
        print("\nInvalid Value Replacement Report:")
        for val, cnt in update_counts.items():
            print(f"Replaced {cnt} instances of {val} with {replacement}.")
    else:
        print("\nNo invalid values found to replace.")
    

# Apply the function to the filtered_df
replace_and_count_invalid_values(filtered_df, invalid_values)

# ------------------------------------------------------
# Verify values are as expected
# ------------------------------------------------------

# Verify specific dates have expected values
test_dates = {
    '1997-06-13 18:06:00': {
        "Altitude": 1410.0,
        "GCLAT": -73.35,
        "GCLON": 320.1,
        "ILAT": 59.16,
        "GLAT": -58.4,
        "GMLT": 14.97,
        "XXLAT": -58.81,
        "XXLON": 15.0,
        "Te1": 6631,
        "Ne1": 779,
        "Pv1": 2.66,
        "Te2": 0,
        "Ne2": 0,
        "Pv2": 0.0,
        "Te3": 0,
        "Ne3": 0,
        "Pv3": 0.0,
        "I1": 21,
        "I2": 0,
        "I3": 0,
    "AL_index_0": -21.0,
    "AL_index_1": -39.0,
    "AL_index_2": -17.0,
    "AL_index_3": -19.0,
    "AL_index_4": -18.0,
    "AL_index_5": -26.0,
    "AL_index_6": -25.0,
    "AL_index_7": -13.0,
    "AL_index_8": -13.0,
    "AL_index_9": -14.0,
    "AL_index_10": -19.0,
    "AL_index_11": -47.0,
    "AL_index_12": -72.0,
    "AL_index_13": -99.0,
    "AL_index_14": -118.0,
    "AL_index_15": -141.0,
    "AL_index_16": -108.0,
    "AL_index_17": -171.0,
    "AL_index_18": -153.0,
    "AL_index_19": -94.0,
    "AL_index_20": -52.0,
    "AL_index_21": -83.0,
    "AL_index_22": -119.0,
    "AL_index_23": -93.0,
    "AL_index_24": -81.0,
    "AL_index_25": -96.0,
    "AL_index_26": -119.0,
    "AL_index_27": -185.0,
    "AL_index_28": -231.0,
    "AL_index_29": -129.0,
    "AL_index_30": -75.0,
    "SYM_H_0": -1.0,
    "SYM_H_1": -3.0,
    "SYM_H_2": -2.0,
    "SYM_H_3": -2.0,
    "SYM_H_4": -2.0,
    "SYM_H_5": 0.0,
    "SYM_H_6": 1.0,
    "SYM_H_7": -1.0,
    "SYM_H_8": -1.0,
    "SYM_H_9": -1.0,
    "SYM_H_10": -1.0,
    "SYM_H_11": 0.0,
    "SYM_H_12": 1.0,
    "SYM_H_13": 1.0,
    "SYM_H_14": 3.0,
    "SYM_H_15": 1.0,
    "SYM_H_16": 1.0,
    "SYM_H_17": 0.0,
    "SYM_H_18": -1.0,
    "SYM_H_19": -4.0,
    "SYM_H_20": -4.0,
    "SYM_H_21": -4.0,
    "SYM_H_22": -3.0,
    "SYM_H_23": -4.0,
    "SYM_H_24": -4.0,
    "SYM_H_25": -1.0,
    "SYM_H_26": -2.0,
    "SYM_H_27": -1.0,
    "SYM_H_28": 1.0,
    "SYM_H_29": -1.0,
    "SYM_H_30": -4.0,
    "f107_index_0": 72.3,
    "f107_index_1": 71.8,
    "f107_index_2": 73.0,
    "f107_index_3": 74.2,
    "Kp_index": 0.0
    }
}

print("\nVerifying values in dataset...")
for date, v in test_dates.items():
    row = filtered_df.loc[date].iloc[0]
    for col, expected_val in v.items():
        actual_val = row[col]
        assert actual_val == expected_val, f"Mismatch for {date} {col}: Expected {expected_val}, got {actual_val}"

# ------------------------------------------------------
# Save by Kp index
# ------------------------------------------------------

# Define the parent directory for all bucketed datasets
parent_output_dir = "output_dataset"
os.makedirs(parent_output_dir, exist_ok=True)

# Get unique Kp index values
unique_kp_values = sorted(filtered_df['Kp_index'].unique())

total_rows_saved = 0
for kp_value in unique_kp_values:
    # Get rows for this Kp value
    kp_df = filtered_df[filtered_df['Kp_index'] == kp_value]
    
    # Prepare the directory within the parent output directory
    output_dir = os.path.join(parent_output_dir, f"akebono_solar_kp_{kp_value}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving {len(kp_df)} rows for Kp index value {kp_value}...")
    
    # Convert to a Dataset and save to disk
    dataset_kp = datasets.Dataset(pa.Table.from_pandas(kp_df))
    dataset_kp.save_to_disk(output_dir)
    
    total_rows_saved += len(kp_df)
    print(f"Kp index {kp_value} saved to: {output_dir}")

# Verify total rows match
assert total_rows_saved == len(filtered_df), (
    f"Total rows saved ({total_rows_saved}) does not match filtered_df size "
    f"({len(filtered_df)})"
)
print(f"Successfully saved all {total_rows_saved} rows")
