import pandas as pd

import matplotlib.pyplot as plt
import os

LOG_DIR = "reward-logs/nn-simulated/invertedpendulum"
PLOT_DIR = f'reward-plots/nn-simulated/invertedpendulum/'
main_df = pd.DataFrame()
max_df = pd.DataFrame()
for directory in os.listdir(LOG_DIR):
    if not directory.startswith("trial"):
        continue
    os.makedirs(os.path.join(PLOT_DIR, directory), exist_ok=True)
    # Read the CSV file
    df = pd.read_csv(os.path.join(LOG_DIR, directory, 'overall_log.txt'))
    main_df = pd.concat([main_df, df], ignore_index=True)
    # print(main_df.shape)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Calculate the actual (non-cumulative) CPU and API times
    df['CPU Time per Iteration'] = df['CPU Time'].diff().fillna(df['CPU Time'].iloc[0])
    df['API Time per Iteration'] = df['API Time'].diff().fillna(df['API Time'].iloc[0])

    # Plot: True Reward and Predicted Reward vs Iteration side by side, plus comparison below
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    # Plot 1: True Reward
    ax1.plot(df['Iteration'], df['True Reward'], marker='o', linestyle='-', linewidth=2, markersize=4, label='Reward', alpha=0.5)

    # Calculate and plot running average (window size of 50)
    window_size = 50
    running_avg = df['True Reward'].rolling(window=window_size, min_periods=1).mean()
    ax1.plot(df['Iteration'], running_avg, linestyle='-', linewidth=2.5, color='red', label=f'Running Average (window={window_size})')

    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('True Reward', fontsize=12)
    ax1.set_title('True Reward vs Iteration', fontsize=14)
    # also add the mean and stddev to the plot
    mean_reward = df['True Reward'].mean()
    stddev_reward = df['True Reward'].std()
    ax1.axhline(mean_reward, color='green', linestyle='--', label='Mean Reward')
    ax1.fill_between(df['Iteration'], mean_reward - stddev_reward, mean_reward + stddev_reward, color='green', alpha=0.2, label='Std Dev')
    # also 
    ax1.text(0.98, 0.02, f'Mean: {mean_reward:.2f}\nStd Dev: {stddev_reward:.2f}', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='bottom', 
             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Predicted Reward
    if 'Predicted Reward' in df.columns:
        ax2.plot(df['Iteration'], df['Predicted Reward'], marker='o', linestyle='-', linewidth=2, markersize=4, label='Predicted Reward', alpha=0.5)

        # Calculate and plot running average (window size of 50)
        running_avg_pred = df['Predicted Reward'].rolling(window=window_size, min_periods=1).mean()
        ax2.plot(df['Iteration'], running_avg_pred, linestyle='-', linewidth=2.5, color='red', label=f'Running Average (window={window_size})')

        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Predicted Reward', fontsize=12)
        ax2.set_title('Predicted Reward vs Iteration', fontsize=14)
        # also add the mean and stddev to the plot
        mean_pred = df['Predicted Reward'].mean()
        stddev_pred = df['Predicted Reward'].std()
        ax2.axhline(mean_pred, color='green', linestyle='--', label='Mean Predicted Reward')
        ax2.fill_between(df['Iteration'], mean_pred - stddev_pred, mean_pred + stddev_pred, color='green', alpha=0.2, label='Std Dev')
        # also 
        ax2.text(0.98, 0.02, f'Mean: {mean_pred:.2f}\nStd Dev: {stddev_pred:.2f}', 
                 transform=ax2.transAxes, fontsize=10, verticalalignment='bottom', 
                 horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Comparison of Rolling Averages
        ax3.plot(df['Iteration'], running_avg, linestyle='-', linewidth=2.5, color='blue', label=f'True Reward Running Avg')
        ax3.plot(df['Iteration'], running_avg_pred, linestyle='-', linewidth=2.5, color='orange', label=f'Predicted Reward Running Avg')
        
        ax3.set_xlabel('Iteration', fontsize=12)
        ax3.set_ylabel('Reward', fontsize=12)
        ax3.set_title('True vs Predicted Reward Running Averages', fontsize=14)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Predicted Reward column not found', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
        ax3.text(0.5, 0.5, 'Predicted Reward column not found', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)

    plt.tight_layout()
    plt.savefig(f'{os.path.join(PLOT_DIR, directory, "reward_vs_iteration")}_{directory}.png', dpi=300)
    plt.show()

    # Store the max reward row in max_df
    max_row = df.loc[df['True Reward'].idxmax()]
    max_df = pd.concat([max_df, max_row.to_frame().T], ignore_index=True)

    print(f"{directory}", df.shape[0], "iterations\n")
print(max_df)
print("Keys in main_df:", main_df.columns)
# Overall summary across all runs
print("\nOverall Summary Statistics Across All Runs:")
# print(f"Average CPU Time per Iteration: {main_df['CPU Time'].diff().mean():.4f} seconds")
# print(f"Average API Time per Iteration: {main_df['API Time'].diff().mean():.4f} seconds")
print(f"Average True Reward: {main_df[' True Reward'].mean():.2f}")
print(f"Standard Deviation of True Reward: {main_df[' True Reward'].std():.2f}")
print(f"Max True Reward: {main_df[' True Reward'].max():.2f}")
print(f"Min True Reward: {main_df[' True Reward'].min():.2f}")

print(f"\n\nAverage True Reward: {max_df['True Reward'].mean():.2f}")
print(f"Standard Deviation of True Reward: {max_df['True Reward'].std():.2f}")
print(f"Max True Reward: {max_df['True Reward'].max():.2f}")
print(f"Min True Reward: {max_df['True Reward'].min():.2f}")

# print("\nMax Reward Summary Across All Runs:")
# print(max_df[['Iteration', ' True Reward', 'CPU Time', 'API Time']])
# print(f"Total tool")
