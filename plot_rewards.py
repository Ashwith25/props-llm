import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_rewards():
    # Define the base directory relative to the script location or absolute
    base_dir = os.path.join(os.getcwd(), 'propsR-log', 'IP_propsR')
    output_dir = os.path.join(os.getcwd(), 'reward-plots/12-21')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Find all trial directories
    # Looking for trial_* folders
    trial_pattern = os.path.join(base_dir, 'trial_*')
    trial_dirs = glob.glob(trial_pattern)
    
    if not trial_dirs:
        print(f"No trial directories found in {base_dir}")
        return

    print(f"Found {len(trial_dirs)} trial directories.")

    for trial_dir in trial_dirs:
        log_file = os.path.join(trial_dir, 'overall_log.txt')
        if not os.path.exists(log_file):
            print(f"No overall_log.txt found in {trial_dir}")
            continue
            
        try:
            print(f"Processing {log_file}...")
            # Read the file. 
            # The header is: Iteration, Params, CPU Time, API Time, Total Episodes, Total Steps, True Reward, Predicted Reward, Confidence
            # We use skipinitialspace=True because there might be spaces after commas
            df = pd.read_csv(log_file, skipinitialspace=True)
            
            # Check if required columns exist
            required_cols = ['Iteration', 'True Reward', 'Predicted Reward', 'Confidence']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Skipping {log_file}: Missing columns {missing_cols}")
                continue

            # Drop rows with missing values in relevant columns
            df.dropna(subset=['True Reward', 'Predicted Reward', 'Confidence'], inplace=True)

            trial_name = os.path.basename(trial_dir)
            
            # Sort by True Reward for the line plot
            df_sorted = df.sort_values(by='True Reward').reset_index(drop=True)

            # --- Plot 1: Line Plot ---
            plt.figure(figsize=(12, 7))
            
            # Define category colors (light backgrounds)
            cat_colors = ['#ffebee', '#e8f5e9', '#e3f2fd', '#f3e5f5'] # Red, Green, Blue, Purple tints
            
            # Helper to find first index >= value
            def find_transition_index(val):
                indices = df_sorted.index[df_sorted['True Reward'] >= val].tolist()
                return indices[0] if indices else None

            # Determine boundaries
            idx_0 = 0
            idx_250 = find_transition_index(250)
            idx_500 = find_transition_index(500)
            idx_750 = find_transition_index(750)
            idx_end = len(df_sorted) - 1

            # Draw background regions
            # 0-25% (0-250)
            end_0_25 = idx_250 if idx_250 is not None else idx_end
            if end_0_25 > idx_0:
                plt.axvspan(idx_0, end_0_25, color=cat_colors[0], alpha=0.3, label='0-25% Range')
            
            # 25-50% (250-500)
            if idx_250 is not None:
                start = idx_250
                end = idx_500 if idx_500 is not None else idx_end
                if end > start:
                    plt.axvspan(start, end, color=cat_colors[1], alpha=0.3, label='25-50% Range')
            
            # 50-75% (500-750)
            if idx_500 is not None:
                start = idx_500
                end = idx_750 if idx_750 is not None else idx_end
                if end > start:
                    plt.axvspan(start, end, color=cat_colors[2], alpha=0.3, label='50-75% Range')

            # 75-100% (750-1000+)
            if idx_750 is not None:
                start = idx_750
                end = idx_end
                if end > start:
                    plt.axvspan(start, end, color=cat_colors[3], alpha=0.3, label='75-100% Range')

            # Plot True Reward
            plt.plot(df_sorted.index, df_sorted['True Reward'], label='True Reward', color='blue', alpha=0.6, linewidth=2)
            
            # Plot Predicted Reward Line
            plt.plot(df_sorted.index, df_sorted['Predicted Reward'], label='Predicted Reward', color='orange', linestyle='--', alpha=0.5)
            
            # Scatter plot for Predicted Reward with Confidence
            # Size scaled by Confidence (1-10) -> 10-100
            sc = plt.scatter(df_sorted.index, df_sorted['Predicted Reward'], 
                             c=df_sorted['Confidence'], cmap='viridis', 
                             s=df_sorted['Confidence'] * 10, 
                             alpha=0.9, label='Predicted (Conf.)', edgecolors='k', linewidth=0.5)
            
            plt.colorbar(sc, label='Confidence (1-10)')
            
            plt.xlabel('Sample Index (Sorted by True Reward)')
            plt.ylabel('Reward')
            plt.title(f'True vs Predicted Reward (Sorted) - {trial_name}')
            
            # Deduplicate legend
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc='upper left')
            
            plt.grid(True, linestyle=':', alpha=0.6)
            
            # Save the line plot
            output_filename = f'IP_propsR_{trial_name}_rewards.png'
            output_path = os.path.join(output_dir, output_filename)
            plt.savefig(output_path)
            plt.close()
            print(f"Saved plot to {output_path}")

            # --- Plot 2: Confusion Matrix ---
            # Define bins based on 0-1000 range
            # 0-25%, 25-50%, 50-75%, 75-100% of 1000
            bins = [-np.inf, 250, 500, 750, np.inf]
            labels = ['0-25%', '25-50%', '50-75%', '75-100%']
            
            # Create categories
            true_cats = pd.cut(df['True Reward'], bins=bins, labels=labels)
            pred_cats = pd.cut(df['Predicted Reward'], bins=bins, labels=labels)
            
            # Calculate confusion matrix
            # We explicitly provide labels to ensure the matrix is 4x4 even if some categories are missing
            cm = confusion_matrix(true_cats, pred_cats, labels=labels)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
            plt.xlabel('Predicted Reward Range')
            plt.ylabel('True Reward Range')
            plt.title(f'Confusion Matrix - {trial_name}')
            
            output_cm_filename = f'IP_propsR_{trial_name}_confusion_matrix.png'
            output_cm_path = os.path.join(output_dir, output_cm_filename)
            plt.savefig(output_cm_path)
            plt.close()
            print(f"Saved confusion matrix to {output_cm_path}")

            # --- Plot 3: Confidence Trend over Iterations ---
            plt.figure(figsize=(12, 6))
            
            # Plot confidence over iterations (not sorted, to see temporal trend)
            plt.plot(df['Iteration'], df['Confidence'], label='Confidence', color='green', alpha=0.3, linewidth=2, marker='o', markersize=1)
            
            # Add a rolling average to see the trend better
            window_size = min(10, len(df) // 5)  # Use 10 or 20% of data, whichever is smaller
            if window_size > 1:
                rolling_avg = df['Confidence'].rolling(window=window_size, center=True).mean()
                plt.plot(df['Iteration'], rolling_avg, label=f'Rolling Avg (window={window_size})', color='darkgreen', alpha=0.8, linewidth=2, linestyle='--')
            
            plt.xlabel('Iteration')
            plt.ylabel('Confidence (1-10)')
            plt.title(f'Confidence Trend over Iterations - {trial_name}')
            plt.legend()
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.ylim(0, 11)  # Set y-axis from 0 to 11 to show the full confidence range
            
            output_conf_filename = f'IP_propsR_{trial_name}_confidence_trend.png'
            output_conf_path = os.path.join(output_dir, output_conf_filename)
            plt.savefig(output_conf_path)
            plt.close()
            print(f"Saved confidence trend to {output_conf_path}")
            
        except Exception as e:
            print(f"Error processing {log_file}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    plot_rewards()
