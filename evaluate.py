from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from models.gaussian import GaussianNetwork
from models.feed_forward import FF_1Network, FF_2Network
from models.transformer import TransformerRegressor
import utils
import json
import matplotlib.pyplot as plt

model_name = 'ff2_8a'
input_columns = ["Altitude", "GCLAT", "GCLON", "ILAT", "GLAT", "GMLT", "XXLAT", "XXLON", "Ne1", "Pv1", "I1", "AL_index_0", "AL_index_1", "AL_index_2", "AL_index_3", "AL_index_4", "AL_index_5", "AL_index_6", "AL_index_7", "AL_index_8", "AL_index_9", "AL_index_10", "AL_index_11", "AL_index_12", "AL_index_13", "AL_index_14", "AL_index_15", "AL_index_16", "AL_index_17", "AL_index_18", "AL_index_19", "AL_index_20", "AL_index_21", "AL_index_22", "AL_index_23", "AL_index_24", "AL_index_25", "AL_index_26", "AL_index_27", "AL_index_28", "AL_index_29", "AL_index_30", "SYM_H_0", "SYM_H_1", "SYM_H_2", "SYM_H_3", "SYM_H_4", "SYM_H_5", "SYM_H_6", "SYM_H_7", "SYM_H_8", "SYM_H_9", "SYM_H_10", "SYM_H_11", "SYM_H_12", "SYM_H_13", "SYM_H_14", "SYM_H_15", "SYM_H_16", "SYM_H_17", "SYM_H_18", "SYM_H_19", "SYM_H_20", "SYM_H_21", "SYM_H_22", "SYM_H_23", "SYM_H_24", "SYM_H_25", "SYM_H_26", "SYM_H_27", "SYM_H_28", "SYM_H_29", "SYM_H_30", "SYM_H_31", "SYM_H_32", "SYM_H_33", "SYM_H_34", "SYM_H_35", "SYM_H_36", "SYM_H_37", "SYM_H_38", "SYM_H_39", "SYM_H_40", "SYM_H_41", "SYM_H_42", "SYM_H_43", "SYM_H_44", "SYM_H_45", "SYM_H_46", "SYM_H_47", "SYM_H_48", "SYM_H_49", "SYM_H_50", "SYM_H_51", "SYM_H_52", "SYM_H_53", "SYM_H_54", "SYM_H_55", "SYM_H_56", "SYM_H_57", "SYM_H_58", "SYM_H_59", "SYM_H_60", "SYM_H_61", "SYM_H_62", "SYM_H_63", "SYM_H_64", "SYM_H_65", "SYM_H_66", "SYM_H_67", "SYM_H_68", "SYM_H_69", "SYM_H_70", "SYM_H_71", "SYM_H_72", "SYM_H_73", "SYM_H_74", "SYM_H_75", "SYM_H_76", "SYM_H_77", "SYM_H_78", "SYM_H_79", "SYM_H_80", "SYM_H_81", "SYM_H_82", "SYM_H_83", "SYM_H_84", "SYM_H_85", "SYM_H_86", "SYM_H_87", "SYM_H_88", "SYM_H_89", "SYM_H_90", "SYM_H_91", "SYM_H_92", "SYM_H_93", "SYM_H_94", "SYM_H_95", "SYM_H_96", "SYM_H_97", "SYM_H_98", "SYM_H_99", "SYM_H_100", "SYM_H_101", "SYM_H_102", "SYM_H_103", "SYM_H_104", "SYM_H_105", "SYM_H_106", "SYM_H_107", "SYM_H_108", "SYM_H_109", "SYM_H_110", "SYM_H_111", "SYM_H_112", "SYM_H_113", "SYM_H_114", "SYM_H_115", "SYM_H_116", "SYM_H_117", "SYM_H_118", "SYM_H_119", "SYM_H_120", "SYM_H_121", "SYM_H_122", "SYM_H_123", "SYM_H_124", "SYM_H_125", "SYM_H_126", "SYM_H_127", "SYM_H_128", "SYM_H_129", "SYM_H_130", "SYM_H_131", "SYM_H_132", "SYM_H_133", "SYM_H_134", "SYM_H_135", "SYM_H_136", "SYM_H_137", "SYM_H_138", "SYM_H_139", "SYM_H_140", "SYM_H_141", "SYM_H_142", "SYM_H_143", "SYM_H_144", "f107_index_0", "f107_index_1", "f107_index_2", "f107_index_3"]
output_column = 'Te1'
columns_to_keep = input_columns + [output_column]
columns_to_normalize = input_columns + [output_column]

# Model
input_size = len(input_columns)
hidden_size = 2048
output_size = 1
model = FF_2Network(input_size, hidden_size, output_size).to("cuda")
model.load_state_dict(torch.load(f'{model_name}.pth'))
model.eval()  # Set the model to evaluation mode

test_df = pd.read_csv('data/test_v5_temporal_split.tsv', sep='\t')
test_df = test_df[columns_to_keep]

# Load means and std from json file
with open(f'data/{model_name}_norm_stats.json', 'r') as f:
    norm_stats = json.load(f)

means = norm_stats['mean']
stds = norm_stats['std']
test_df_norm = utils.normalize_df(test_df, means, stds, columns_to_normalize)
test_ds = utils.DataFrameDataset(test_df_norm, input_columns, output_column)
test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=8)

target_mean = means[output_column]
target_std = stds[output_column]

predictions, true_values = [], []

with torch.no_grad():
    for x, y in tqdm(test_loader, desc="Evaluating"):
        x = x.to("cuda")
        y = y.to("cuda")
        
        # Forward pass
        if model.__class__.__name__ == 'GaussianNetwork':
            y_pred, var = model(x)
        else:
            y_pred = model(x)

        # Unnormalize predictions and true values
        y_true = utils.unnormalize_mean(y.cpu(), target_mean, target_std)
        y_pred = utils.unnormalize_mean(y_pred.cpu(), target_mean, target_std)
        
        predictions.extend(y_pred.flatten().tolist())
        true_values.extend(y_true.flatten().tolist())

# Calculate deviations
deviations = [pred - true for pred, true in zip(predictions, true_values)]

# Calculate percentages within specified absolute deviations
thresholds = [100, 200, 300, 500, 1000, 2000, 5000]
percentages = [
    sum(abs(dev) <= threshold for dev in deviations) / len(deviations) * 100
    for threshold in thresholds
]

# Calculate percentages within specified relative deviations
relative_thresholds = [5, 10, 15, 20]
relative_percentages = [
    sum(abs(dev) / true * 100 <= threshold for dev, true in zip(deviations, true_values)) / len(deviations) * 100
    for threshold in relative_thresholds
]

# Plot histogram
plt.figure(figsize=(12, 8))
plt.hist(deviations, bins=50, edgecolor='black')
plt.xlabel('Deviation from Ground Truth')
plt.ylabel('Frequency')
plt.title('Distribution of Model Predictions Deviation')

# Add text box with percentages
text = "\n".join([
    f"Within {threshold}: {percentage:.2f}%"
    for threshold, percentage in zip(thresholds, percentages)
] + ["\n"] + [  # Add an empty line between absolute and relative thresholds
    f"Within {threshold}%: {percentage:.2f}%"
    for threshold, percentage in zip(relative_thresholds, relative_percentages)
])
plt.text(0.95, 0.95, text, transform=plt.gca().transAxes, 
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()

# Save the plot
plt.savefig(f'{model_name}_deviation.png')
plt.close()  # Close the figure to free up memory