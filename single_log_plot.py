import pandas as pd

import matplotlib.pyplot as plt
import os

LOG_DIR = "InvertedDoublePendulum_gpt-oss_120b_propsp_using_tools_bias"

os.makedirs(f'plots/{LOG_DIR}', exist_ok=True)
# Read the CSV file
df = pd.read_csv(os.path.join(f"logs/{LOG_DIR}", 'overall_log.txt'))
# main_df = pd.concat([main_df, df], ignore_index=True)
# print(main_df.shape)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Calculate the actual (non-cumulative) CPU and API times
df['CPU Time per Iteration'] = df['CPU Time'].diff().fillna(df['CPU Time'].iloc[0])
df['API Time per Iteration'] = df['API Time'].diff().fillna(df['API Time'].iloc[0])

# Plot 1: Total Reward vs Iteration with Running Average
plt.figure(figsize=(10, 6))
plt.plot(df['Iteration'], df['Total Reward'], marker='o', linestyle='-', linewidth=2, markersize=4, label='Reward', alpha=0.5)

# Calculate and plot running average (window size of 5)
window_size = 50
running_avg = df['Total Reward'].rolling(window=window_size, min_periods=1).mean()
plt.plot(df['Iteration'], running_avg, linestyle='-', linewidth=2.5, color='red', label=f'Running Average (window={window_size})')

plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Total Reward', fontsize=12)
plt.title('Total Reward vs Iteration', fontsize=14)
# also add the mean and stddev to the plot
mean_reward = df['Total Reward'].mean()
stddev_reward = df['Total Reward'].std()
plt.axhline(mean_reward, color='green', linestyle='--', label='Mean Reward')
plt.fill_between(df['Iteration'], mean_reward - stddev_reward, mean_reward + stddev_reward, color='green', alpha=0.2, label='Std Dev')
# also 
plt.text(0.98, 0.02, f'Mean: {mean_reward:.2f}\nStd Dev: {stddev_reward:.2f}', 
            transform=plt.gca().transAxes, fontsize=10, verticalalignment='bottom', 
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'plots/{LOG_DIR}/reward_vs_iteration.png', dpi=300)
plt.show()

# # Plot 2: CPU Time and API Time vs Rewards (dual y-axes)
# fig, ax1 = plt.subplots(figsize=(10, 6))

# # Plot CPU Time on left y-axis
# color = 'tab:blue'
# ax1.set_xlabel('Total Reward', fontsize=12)
# ax1.set_ylabel('CPU Time per Iteration (seconds)', fontsize=12, color=color)
# ax1.scatter(df['Total Reward'], df['CPU Time per Iteration'], alpha=0.6, s=50, color=color, label='CPU Time')
# ax1.tick_params(axis='y', labelcolor=color)
# ax1.grid(True, alpha=0.3)

# # Create second y-axis for API Time
# ax2 = ax1.twinx()
# color = 'tab:orange'
# ax2.set_ylabel('API Time per Iteration (seconds)', fontsize=12, color=color)
# ax2.scatter(df['Total Reward'], df['API Time per Iteration'], alpha=0.6, s=50, color=color, label='API Time')
# ax2.tick_params(axis='y', labelcolor=color)

# # Add title and legends
# plt.title('CPU Time and API Time vs Total Reward', fontsize=14)
# fig.legend(loc='upper left', bbox_to_anchor=(0.15, 0.95))
# plt.tight_layout()
# plt.savefig(f'plots/idp_propsp_loop_50_traj_high/time_vs_reward_{directory}.png', dpi=300)
# plt.show()

# print("Plots saved successfully!")
print(f"\nSummary Statistics:")
print(f"Average CPU Time per Iteration: {df['CPU Time per Iteration'].mean():.4f} seconds")
print(f"Average API Time per Iteration: {df['API Time per Iteration'].mean():.4f} seconds")
print(f"Average Total Reward: {df['Total Reward'].mean():.2f}")
print(f"Standard Deviation of Total Reward: {df['Total Reward'].std():.2f}")
print(f"Max Total Reward: {df['Total Reward'].max():.2f}")
print(f"Min Total Reward: {df['Total Reward'].min():.2f}")