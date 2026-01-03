path = "reward-logs/nn-simulated/invertedpendulum/trial_1/overall_log.txt"

import pandas as pd
import numpy as np
import io

data = {}
with open(path, 'r') as f:
    lines = f.readlines()
    num_lines = len(lines)
    part_size = num_lines // 4
    for i in range(4):
        part_lines = lines[i*part_size : (i+1)*part_size] if i < 3 else lines[i*part_size :]
        df = pd.read_csv(io.StringIO(''.join(part_lines)), header=None)
        df.columns = [' Iteration', ' Params', ' CPU Time', ' API Time', ' Total Episodes', ' Total Steps', ' True Reward', ' Predicted Reward', ' Confidence']
        # Convert numeric columns to float
        df[' True Reward'] = pd.to_numeric(df[' True Reward'], errors='coerce')
        df[' Predicted Reward'] = pd.to_numeric(df[' Predicted Reward'], errors='coerce')
        df[' Confidence'] = pd.to_numeric(df[' Confidence'], errors='coerce')
        data[f'sample_{i*part_size}_{(i+1)*part_size}'] = df

print("\n" + "="*70)
tolerance = 30
print("SUMMARY STATISTICS with tolerance of", tolerance)
print("="*70)
print(f"{'Sample':<15}\t | {'MAE':<8} | {'RMSE':<8} | {'Corr':<8} | {'Avg Conf':<10} | {'Accuracy':<10}")
print("-"*70)
for trial in list(data.keys()):
    df = data[trial]
    error = df[' Predicted Reward'] - df[' True Reward']
    abs_error = np.abs(error)
    
    mae = abs_error.mean()
    rmse = np.sqrt((error**2).mean())
    corr = df[' True Reward'].corr(df[' Predicted Reward'])
    avg_conf = df[' Confidence'].mean()
    
    # Calculate accuracy within a tolerance of 10
    
    accurate_predictions = np.abs(error) <= tolerance
    accuracy = accurate_predictions.sum() / len(df) * 100  # percentage
    
    print(f"{trial:<10} \t | {mae:<8.2f} | {rmse:<8.2f} | {corr:<8.3f} | {avg_conf:<10.3f} | {accuracy:<10.1f}%")