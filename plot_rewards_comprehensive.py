import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Base path for propsR logs
base_path = '/scratch/apoojar4/propsR-sim-log/MCD_propsR'
pdf_path = 'reward-plots/propsR_sim_log/MCD_propsR/propsR_comprehensive_analysis.pdf'

trials = ['trial_1', 'trial_2', 'trial_3']

# Load data from all trials
data = {}
for trial in trials:
    log_file = os.path.join(base_path, trial, 'overall_log.txt')
    if os.path.exists(log_file):
        data[trial] = pd.read_csv(log_file)
    else:
        print(f"Warning: {log_file} not found")

print(f"Loaded {len(data)} trials")

# Calculate reward ranges dynamically
all_rewards = []
for df in data.values():
    all_rewards.extend(df[' True Reward'].values)

min_reward = min(all_rewards)
max_reward = max(all_rewards)
range_size = (max_reward - min_reward) / 4

print(f"Reward Range: Min={min_reward:.2f}, Max={max_reward:.2f}, Range Size={range_size:.2f}")

# Define reward categories based on dynamic ranges
def categorize_reward(reward):
    if reward <= min_reward + range_size:
        return 'Very Low'
    elif reward <= min_reward + 2 * range_size:
        return 'Low'
    elif reward <= min_reward + 3 * range_size:
        return 'Medium'
    else:
        return 'High'


categories = ['Very Low', 'Low', 'Medium', 'High']

# Create PDF with all plots
with PdfPages(pdf_path) as pdf:
    
    # ===== PAGE 1: Scatter Plots for first 4 trials =====
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Scatter: True Reward vs Predicted Reward (Trials 1-4)', fontsize=16, fontweight='bold', y=0.995)
    axes = axes.flatten()
    
    trial_list = list(data.keys())[:4]
    for idx, trial in enumerate(trial_list):
        df = data[trial]
        ax = axes[idx]
        
        ax.scatter(df[' True Reward'], df[' Predicted Reward'], alpha=0.6, s=50, color='steelblue')
        min_val = min(df[' True Reward'].min(), df[' Predicted Reward'].min())
        max_val = max(df[' True Reward'].max(), df[' Predicted Reward'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        corr = df[' True Reward'].corr(df[' Predicted Reward'])
        mae = np.abs(df[' Predicted Reward'] - df[' True Reward']).mean()
        
        ax.set_xlabel('True Reward', fontsize=10)
        ax.set_ylabel('Predicted Reward', fontsize=10)
        ax.set_title(f'{trial} - Corr: {corr:.3f}, MAE: {mae:.2f}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # ===== PAGE 2: Absolute Error Plots for first 4 trials =====
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Absolute Prediction Error Over Iterations (Trials 1-4)', fontsize=16, fontweight='bold', y=0.995)
    axes = axes.flatten()
    
    for idx, trial in enumerate(trial_list):
        df = data[trial]
        ax = axes[idx]
        
        abs_error = np.abs(df[' Predicted Reward'] - df[' True Reward'])
        ax.plot(df['Iteration'], abs_error, color='orange', linewidth=1.5, marker='s', markersize=3, alpha=0.7)
        ax.set_xlabel('Iteration', fontsize=10)
        ax.set_ylabel('Absolute Error', fontsize=10)
        
        mean_abs_error = abs_error.mean()
        ax.axhline(y=mean_abs_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_abs_error:.2f}')
        ax.set_title(f'{trial} - Absolute Error', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # ===== PAGE 3: Confusion Matrices for first 4 trials =====
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Confusion Matrices: Reward Category Classification (Trials 1-4)', fontsize=16, fontweight='bold', y=0.995)
    axes = axes.flatten()
    
    for idx, trial in enumerate(trial_list):
        df = data[trial]
        ax = axes[idx]
        
        true_categories = df[' True Reward'].apply(categorize_reward)
        pred_categories = df[' Predicted Reward'].apply(categorize_reward)
        cm = confusion_matrix(true_categories, pred_categories, labels=categories)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    xticklabels=categories, yticklabels=categories, cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted Category', fontsize=10, fontweight='bold')
        ax.set_ylabel('True Category', fontsize=10, fontweight='bold')
        ax.set_title(f'{trial}', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # ===== PAGE 4: Normalized Confusion Matrices for first 4 trials =====
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Normalized Confusion Matrices - Percentage (Trials 1-4)', fontsize=16, fontweight='bold', y=0.995)
    axes = axes.flatten()
    
    for idx, trial in enumerate(trial_list):
        df = data[trial]
        ax = axes[idx]
        
        true_categories = df[' True Reward'].apply(categorize_reward)
        pred_categories = df[' Predicted Reward'].apply(categorize_reward)
        cm = confusion_matrix(true_categories, pred_categories, labels=categories)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax,
                    xticklabels=categories, yticklabels=categories, 
                    cbar_kws={'label': 'Percentage (%)'}, vmin=0, vmax=100)
        ax.set_xlabel('Predicted Category', fontsize=10, fontweight='bold')
        ax.set_ylabel('True Category', fontsize=10, fontweight='bold')
        ax.set_title(f'{trial}', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # ===== PAGE 5: Confidence Trends for first 4 trials =====
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Confidence Trend Over Iterations (Trials 1-4)', fontsize=16, fontweight='bold', y=0.995)
    axes = axes.flatten()
    
    for idx, trial in enumerate(trial_list):
        df = data[trial]
        ax = axes[idx]
        
        ax.plot(df['Iteration'], df[' Confidence'], color='green', linewidth=2, marker='o', markersize=3, alpha=0.7)
        ax.fill_between(df['Iteration'], df[' Confidence'], alpha=0.2, color='green')
        
        mean_conf = df[' Confidence'].mean()
        ax.axhline(y=mean_conf, color='darkgreen', linestyle='--', linewidth=2, label=f'Mean: {mean_conf:.3f}')
        
        ax.set_xlabel('Iteration', fontsize=10)
        ax.set_ylabel('Confidence', fontsize=10)
        ax.set_title(f'{trial}', fontsize=11, fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # ===== PAGE 6: Statistics Summary =====
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance Statistics Summary (All Trials)', fontsize=16, fontweight='bold', y=0.995)
    
    # Prepare statistics
    stats = {}
    for trial, df in data.items():
        error = df[' Predicted Reward'] - df[' True Reward']
        abs_error = np.abs(error)
        stats[trial] = {
            'MAE': abs_error.mean(),
            'RMSE': np.sqrt((error**2).mean()),
            'Correlation': df[' True Reward'].corr(df[' Predicted Reward']),
            'Accuracy': 0  # Will be calculated below
        }
    
    trial_names = list(stats.keys())
    
    # Plot 1: MAE and RMSE comparison
    ax1 = axes[0, 0]
    mae_values = [stats[t]['MAE'] for t in trial_names]
    rmse_values = [stats[t]['RMSE'] for t in trial_names]
    x_pos = np.arange(len(trial_names))
    width = 0.35
    
    ax1.bar(x_pos - width/2, mae_values, width, label='MAE', color='steelblue', alpha=0.8)
    ax1.bar(x_pos + width/2, rmse_values, width, label='RMSE', color='orange', alpha=0.8)
    ax1.set_xlabel('Trial', fontsize=11)
    ax1.set_ylabel('Error', fontsize=11)
    ax1.set_title('Mean Absolute Error vs RMSE', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(trial_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Correlation comparison
    ax2 = axes[0, 1]
    corr_values = [stats[t]['Correlation'] for t in trial_names]
    colors = ['green' if c > 0.5 else 'orange' if c > 0 else 'red' for c in corr_values]
    bars = ax2.bar(trial_names, corr_values, color=colors, alpha=0.8)
    ax2.set_ylabel('Correlation Coefficient', fontsize=11)
    ax2.set_title('Correlation: True vs Predicted', fontsize=12, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylim([min(corr_values) - 0.1, 1])
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, corr_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{val:.3f}', 
                ha='center', va='bottom', fontsize=9)
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Average Confidence
    ax3 = axes[1, 0]
    conf_values = [data[t][' Confidence'].mean() for t in trial_names]
    bars = ax3.bar(trial_names, conf_values, color='purple', alpha=0.7)
    ax3.set_ylabel('Average Confidence', fontsize=11)
    ax3.set_title('Mean Confidence Across Trials', fontsize=12, fontweight='bold')
    ax3.set_ylim([0, 1.0])
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, conf_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height, f'{val:.3f}', 
                ha='center', va='bottom', fontsize=9)
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Classification Accuracy
    ax4 = axes[1, 1]
    accuracy_values = []
    for trial in trial_names:
        df = data[trial]
        true_categories = df[' True Reward'].apply(categorize_reward)
        pred_categories = df[' Predicted Reward'].apply(categorize_reward)
        cm = confusion_matrix(true_categories, pred_categories, labels=categories)
        accuracy = np.trace(cm) / cm.sum() * 100
        accuracy_values.append(accuracy)
    
    bars = ax4.bar(trial_names, accuracy_values, color='teal', alpha=0.7)
    ax4.set_ylabel('Accuracy (%)', fontsize=11)
    ax4.set_title('Category Classification Accuracy', fontsize=12, fontweight='bold')
    ax4.set_ylim([0, 100])
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, accuracy_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height, f'{val:.1f}%', 
                ha='center', va='bottom', fontsize=9)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # ===== PAGE 7: Trial 5 Detailed Views =====
    if 'trial_5' in data:
        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        fig.suptitle('Trial 5 - Detailed Analysis', fontsize=16, fontweight='bold')
        
        df_t5 = data['trial_5']
        
        # Scatter plot
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(df_t5[' True Reward'], df_t5[' Predicted Reward'], alpha=0.6, s=50, color='steelblue')
        min_val = min(df_t5[' True Reward'].min(), df_t5[' Predicted Reward'].min())
        max_val = max(df_t5[' True Reward'].max(), df_t5[' Predicted Reward'].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        corr_t5 = df_t5[' True Reward'].corr(df_t5[' Predicted Reward'])
        mae_t5 = np.abs(df_t5[' Predicted Reward'] - df_t5[' True Reward']).mean()
        ax1.set_xlabel('True Reward', fontsize=10)
        ax1.set_ylabel('Predicted Reward', fontsize=10)
        ax1.set_title(f'Scatter Plot - Corr: {corr_t5:.3f}, MAE: {mae_t5:.2f}', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Error trend
        ax2 = fig.add_subplot(gs[0, 1])
        abs_error_t5 = np.abs(df_t5[' Predicted Reward'] - df_t5[' True Reward'])
        ax2.plot(df_t5['Iteration'], abs_error_t5, color='orange', linewidth=1.5, marker='s', markersize=3, alpha=0.7)
        ax2.axhline(y=abs_error_t5.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {abs_error_t5.mean():.2f}')
        ax2.set_xlabel('Iteration', fontsize=10)
        ax2.set_ylabel('Absolute Error', fontsize=10)
        ax2.set_title('Absolute Error Trend', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Confusion matrix
        ax3 = fig.add_subplot(gs[1, 0])
        true_cat_t5 = df_t5[' True Reward'].apply(categorize_reward)
        pred_cat_t5 = df_t5[' Predicted Reward'].apply(categorize_reward)
        cm_t5 = confusion_matrix(true_cat_t5, pred_cat_t5, labels=categories)
        sns.heatmap(cm_t5, annot=True, fmt='d', cmap='Blues', ax=ax3,
                    xticklabels=categories, yticklabels=categories, cbar_kws={'label': 'Count'})
        ax3.set_xlabel('Predicted Category', fontsize=10, fontweight='bold')
        ax3.set_ylabel('True Category', fontsize=10, fontweight='bold')
        ax3.set_title('Confusion Matrix', fontsize=11, fontweight='bold')
        
        # Confidence
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(df_t5['Iteration'], df_t5[' Confidence'], color='green', linewidth=2, marker='o', markersize=3, alpha=0.7)
        ax4.fill_between(df_t5['Iteration'], df_t5[' Confidence'], alpha=0.2, color='green')
        mean_conf_t5 = df_t5[' Confidence'].mean()
        ax4.axhline(y=mean_conf_t5, color='darkgreen', linestyle='--', linewidth=2, label=f'Mean: {mean_conf_t5:.3f}')
        ax4.set_xlabel('Iteration', fontsize=10)
        ax4.set_ylabel('Confidence', fontsize=10)
        ax4.set_title('Confidence Trend', fontsize=11, fontweight='bold')
        ax4.set_ylim([0, 1.0])
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Normalized confusion matrix
        ax5 = fig.add_subplot(gs[2, 0])
        cm_t5_norm = cm_t5.astype('float') / cm_t5.sum(axis=1)[:, np.newaxis] * 100
        sns.heatmap(cm_t5_norm, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax5,
                    xticklabels=categories, yticklabels=categories, 
                    cbar_kws={'label': 'Percentage (%)'}, vmin=0, vmax=100)
        ax5.set_xlabel('Predicted Category', fontsize=10, fontweight='bold')
        ax5.set_ylabel('True Category', fontsize=10, fontweight='bold')
        ax5.set_title('Normalized Confusion Matrix', fontsize=11, fontweight='bold')
        
        # Reward distribution comparison
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.hist(df_t5[' True Reward'], bins=20, alpha=0.6, label='True Reward', color='steelblue', edgecolor='black')
        ax6.hist(df_t5[' Predicted Reward'], bins=20, alpha=0.6, label='Predicted Reward', color='orange', edgecolor='black')
        ax6.set_xlabel('Reward Value', fontsize=10)
        ax6.set_ylabel('Frequency', fontsize=10)
        ax6.set_title('Reward Distribution Comparison', fontsize=11, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    # ===== PAGE 8: All Trials Overlay Comparison =====
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('All Trials Overlay Comparison', fontsize=16, fontweight='bold', y=0.995)
    
    # Absolute error overlay
    ax1 = axes[0, 0]
    for trial, df in data.items():
        abs_error = np.abs(df[' Predicted Reward'] - df[' True Reward'])
        ax1.plot(df['Iteration'], abs_error, linewidth=1.5, marker='o', markersize=2, alpha=0.6, label=trial)
    ax1.set_xlabel('Iteration', fontsize=10)
    ax1.set_ylabel('Absolute Error', fontsize=10)
    ax1.set_title('Absolute Error - All Trials', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Confidence overlay
    ax2 = axes[0, 1]
    for trial, df in data.items():
        ax2.plot(df['Iteration'], df[' Confidence'], linewidth=1.5, marker='o', markersize=2, alpha=0.6, label=trial)
    ax2.set_xlabel('Iteration', fontsize=10)
    ax2.set_ylabel('Confidence', fontsize=10)
    ax2.set_title('Confidence - All Trials', fontsize=11, fontweight='bold')
    ax2.set_ylim([0, 1.0])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right')
    
    # Error distribution comparison
    ax3 = axes[1, 0]
    error_data = [np.abs(data[t][' Predicted Reward'] - data[t][' True Reward']).values for t in trial_names]
    ax3.boxplot(error_data, labels=trial_names)
    ax3.set_ylabel('Absolute Error', fontsize=10)
    ax3.set_title('Error Distribution Across Trials', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Confidence distribution comparison
    ax4 = axes[1, 1]
    conf_data = [data[t][' Confidence'].values for t in trial_names]
    ax4.boxplot(conf_data, labels=trial_names)
    ax4.set_ylabel('Confidence', fontsize=10)
    ax4.set_title('Confidence Distribution Across Trials', fontsize=11, fontweight='bold')
    ax4.set_ylim([0, 1.0])
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

print(f"\n{'='*70}")
print(f"Comprehensive PDF saved to: {pdf_path}")
print(f"{'='*70}")

# Print summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
print(f"{'Trial':<10} | {'MAE':<8} | {'RMSE':<8} | {'Corr':<8} | {'Avg Conf':<10} | {'Accuracy':<10}")
print("-"*70)

for trial in list(data.keys()):
    df = data[trial]
    error = df[' Predicted Reward'] - df[' True Reward']
    abs_error = np.abs(error)
    
    mae = abs_error.mean()
    rmse = np.sqrt((error**2).mean())
    corr = df[' True Reward'].corr(df[' Predicted Reward'])
    avg_conf = df[' Confidence'].mean()
    
    true_categories = df[' True Reward'].apply(categorize_reward)
    pred_categories = df[' Predicted Reward'].apply(categorize_reward)
    cm = confusion_matrix(true_categories, pred_categories, labels=categories)
    accuracy = np.trace(cm) / cm.sum() * 100
    
    print(f"{trial:<10} | {mae:<8.2f} | {rmse:<8.2f} | {corr:<8.3f} | {avg_conf:<10.3f} | {accuracy:<10.1f}%")

print("="*70)
print("PDF generation completed successfully!")
print("="*70)