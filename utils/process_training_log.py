"""
Script to extract iteration per second (it/s) values from DLRM training log and compute average recommendations per second (rec/s) 
"""

import re
import numpy as np
import os
import csv
import argparse
import matplotlib.pyplot as plt

def load_config() -> dict:
    config = {}
    with open('./training_config.sh', "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "#" in line:
                line = line.split("#", 1)[0].strip()
            if line.startswith("export "):
                line = line.replace("export ", "", 1)
            if "=" in line:
                key, value = line.split("=", 1)
                config[key.strip()] = value.strip()
    return config

# Set up argument parser
parser = argparse.ArgumentParser(description='Process training log to extract it/s values and calculate rec/s.')
parser.add_argument('--log_file', help='Path to the log file')
parser.add_argument('--world_size', type=int, default=8, help='World size')
parser.add_argument('--num_samples', type=int, default=20, help='Number of samples to consider for averaging rec/s')
parser.add_argument('--plot_loss', action='store_true', help='Flag to plot training loss')

# Parse arguments
args = parser.parse_args()

# Path to files
csv_file_path = 'results.csv'
log_file_path = args.log_file
world_size = args.world_size

# Read batch size from config file
config = load_config()
batch_size = int(config['BATCH_SIZE'])

# Extract timestamp from log file name
timestamp_pattern = re.compile(r'log_(\d{4}:\d{2}:\d{2}_\d{2}:\d{2}:\d{2})\.txt')
timestamp_match = timestamp_pattern.search(log_file_path)
timestamp = timestamp_match.group(1) if timestamp_match else 'Unknown'

# Lists to store the values
iteration_per_s_list = []
mean_loss_values = []

# Regular expressions
its_pattern = re.compile(r'\b(\d+\.\d+)it/s\b')
sit_pattern = re.compile(r'\b(\d+\.\d+)s/it\b')
loss_pattern = re.compile(r'Mean loss:\s*(\d+\.\d+)')

# Parse Log File
with open(log_file_path, 'r') as file:
    for line in file:
        # Extract performance, loss values
        if 'Epoch' in line:
            match_its = its_pattern.search(line)
            match_sit = sit_pattern.search(line)
            if match_its:
                iteration_per_s_list.append(float(match_its.group(1)))
            elif match_sit:
                iteration_per_s_list.append(1.0/float(match_sit.group(1)))
        
        loss_match = loss_pattern.search(line)
        if loss_match:
            mean_loss_values.append(float(loss_match.group(1)))

# Compute rec/s values
num_nodes = args.world_size // 8
rec_per_s = np.array(iteration_per_s_list) * batch_size

if len(rec_per_s) > 1:
    mean_rec_per_s = np.mean(rec_per_s[-args.num_samples:-1]) if len(rec_per_s) > args.num_samples else np.mean(rec_per_s[:-1])
    std_rec_per_s = np.std(rec_per_s[-args.num_samples:-1]) if len(rec_per_s) > args.num_samples else np.std(rec_per_s[:-1])
    coeff_of_var = std_rec_per_s / (mean_rec_per_s + 1e-3)
else:
    mean_rec_per_s = rec_per_s[0] if len(rec_per_s) > 0 else 0
    std_rec_per_s = 0
    coeff_of_var = 0
    
# Append results to CSV file
file_exists = os.path.isfile(csv_file_path)
fieldnames = ['Time Stamp', 'Num. Nodes', 'Batch Size', 'Recommendations/s (mean)', 'Recommendations/s (std/mean)']

with open(csv_file_path, mode='a', newline='') as csv_file:

    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()

    writer.writerow({
        'Time Stamp': timestamp,
        'Num. Nodes': num_nodes, 
        'Batch Size': batch_size, 
        'Recommendations/s (mean)': f"{mean_rec_per_s:.2f}", 
        'Recommendations/s (std/mean)': f"{coeff_of_var:.2f}"
    })

print(f"Results appended to: {csv_file_path}")

# Create training loss plots 
if args.plot_loss:
    plot_folder = './loss_analysis_logs'
    os.makedirs(plot_folder, exist_ok=True)

    if len(mean_loss_values) > 0:
        fig_loss, ax_loss = plt.subplots(1, 1, figsize=(12, 6))
        ax_loss.plot(range(len(mean_loss_values)), mean_loss_values, 'r-', linewidth=2, marker='o', markersize=4, label='Mean Loss')
        ax_loss.set_xlabel('Training Step')
        ax_loss.set_ylabel('Mean Loss')
        ax_loss.set_title(f'Training Loss - {timestamp}\nNodes: {num_nodes}, Batch Size: {batch_size}')
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_folder, f'training_loss_bs_{batch_size}_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig_loss)
