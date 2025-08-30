import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging(log_dir, experiment_name):
    """
    Setup comprehensive logging for experiments
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create unique log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/{experiment_name}_{timestamp}.log"
    
    # Configure logging with UTF-8 encoding
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(experiment_name)
    logger.info(f"Logging initialized for experiment: {experiment_name}")
    logger.info(f"Log file: {log_file}")
    
    return logger, log_file

def save_experiment_config(config_dict, save_path):
    """
    Save experiment configuration to JSON file
    """
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=4, default=str)

def calculate_class_weights(y_train):
    """
    Calculate class weights for imbalanced dataset
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        'balanced',
        classes=classes,
        y=y_train
    )
    
    weight_dict = dict(zip(classes, class_weights))
    return weight_dict

def evaluate_model(model, X_test, y_test, class_names=None, save_dir=None):
    """
    Comprehensive model evaluation with visualizations
    """
    # Make predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Classification report
    report = classification_report(y_test, y_pred, 
                                 target_names=class_names, 
                                 output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate additional metrics
    if len(np.unique(y_test)) == 2:  # Binary classification
        auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
    else:  # Multi-class
        auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    
    results = {
        'classification_report': report,
        'confusion_matrix': cm,
        'auc_score': auc_score,
        'accuracy': report['accuracy'],
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score']
    }
    
    # Create visualizations
    if save_dir:
        plot_confusion_matrix(cm, class_names, save_dir)
        plot_classification_metrics(report, save_dir)
    
    return results

def plot_confusion_matrix(cm, class_names, save_dir):
    """
    Plot and save confusion matrix
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_classification_metrics(report, save_dir):
    """
    Plot classification metrics
    """
    # Extract metrics for each class
    classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    metrics = ['precision', 'recall', 'f1-score']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, metric in enumerate(metrics):
        values = [report[cls][metric] for cls in classes]
        bars = axes[i].bar(classes, values, color=plt.cm.Set3(np.linspace(0, 1, len(classes))))
        axes[i].set_title(f'{metric.capitalize()}', fontsize=14, fontweight='bold')
        axes[i].set_ylabel(metric.capitalize(), fontsize=12)
        axes[i].set_ylim(0, 1)
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/classification_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_history(history, save_dir):
    """
    Plot training history
    """
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'auc']
    available_metrics = [m for m in metrics if m in history.history]
    
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=(15, 10))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    for i, metric in enumerate(available_metrics):
        if i < len(axes):
            axes[i].plot(history.history[metric], label=f'Training {metric}', linewidth=2)
            
            val_metric = f'val_{metric}'
            if val_metric in history.history:
                axes[i].plot(history.history[val_metric], label=f'Validation {metric}', linewidth=2)
            
            axes[i].set_title(f'{metric.capitalize()}', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Epoch', fontsize=12)
            axes[i].set_ylabel(metric.capitalize(), fontsize=12)
            axes[i].legend(fontsize=10)
            axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(available_metrics), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_experiment_summary(results, config, save_path):
    """
    Create a comprehensive experiment summary
    """
    summary = {
        'experiment_timestamp': datetime.now().isoformat(),
        'configuration': config,
        'results': {
            'accuracy': results['accuracy'],
            'auc_score': results['auc_score'],
            'macro_f1': results['macro_f1'],
            'weighted_f1': results['weighted_f1']
        },
        'detailed_results': results
    }
    
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=4, default=str)
    
    return summary

class ExperimentTracker:
    """
    Track multiple experiments and compare results
    """
    
    def __init__(self, base_dir='experiments'):
        self.base_dir = base_dir
        self.experiments = []
        os.makedirs(base_dir, exist_ok=True)
    
    def add_experiment(self, name, results, config):
        """
        Add a new experiment to tracking
        """
        experiment = {
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'config': config
        }
        self.experiments.append(experiment)
        
        # Save to file
        self.save_experiments()
    
    def save_experiments(self):
        """
        Save all experiments to JSON file
        """
        with open(f'{self.base_dir}/experiments_log.json', 'w') as f:
            json.dump(self.experiments, f, indent=4, default=str)
    
    def load_experiments(self):
        """
        Load experiments from JSON file
        """
        try:
            with open(f'{self.base_dir}/experiments_log.json', 'r') as f:
                self.experiments = json.load(f)
        except FileNotFoundError:
            self.experiments = []
    
    def compare_experiments(self, metrics=['accuracy', 'auc_score', 'macro_f1']):
        """
        Compare experiments across different metrics
        """
        if not self.experiments:
            print("No experiments to compare")
            return
        
        # Create comparison DataFrame
        comparison_data = []
        for exp in self.experiments:
            row = {'Experiment': exp['name'], 'Timestamp': exp['timestamp']}
            for metric in metrics:
                row[metric] = exp['results'].get(metric, np.nan)
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Plot comparison
        fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 6))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            bars = axes[i].bar(df['Experiment'], df[metric], 
                              color=plt.cm.Set3(np.linspace(0, 1, len(df))))
            axes[i].set_title(f'{metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            axes[i].set_ylabel(metric.replace("_", " ").title(), fontsize=12)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, df[metric]):
                if not np.isnan(value):
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.base_dir}/experiments_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return df

def create_progressive_training_visualization(stage_results, save_dir):
    """
    Visualize progressive training results across stages
    """
    stages = list(range(len(stage_results)))
    accuracies = [result['accuracy'] for result in stage_results]
    f1_scores = [result['macro_f1'] for result in stage_results]
    auc_scores = [result['auc_score'] for result in stage_results]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Accuracy progression
    ax1.plot(stages, accuracies, 'o-', linewidth=3, markersize=8, color='#FF6B6B')
    ax1.set_title('Accuracy Progression', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Training Stage', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # F1-Score progression
    ax2.plot(stages, f1_scores, 's-', linewidth=3, markersize=8, color='#4ECDC4')
    ax2.set_title('F1-Score Progression', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Training Stage', fontsize=12)
    ax2.set_ylabel('Macro F1-Score', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # AUC progression
    ax3.plot(stages, auc_scores, '^-', linewidth=3, markersize=8, color='#45B7D1')
    ax3.set_title('AUC Progression', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Training Stage', fontsize=12)
    ax3.set_ylabel('AUC Score', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Add annotations for final values
    for ax, values, label in zip([ax1, ax2, ax3], 
                                [accuracies, f1_scores, auc_scores],
                                ['Accuracy', 'F1-Score', 'AUC']):
        final_value = values[-1]
        ax.annotate(f'Final: {final_value:.3f}',
                   xy=(stages[-1], final_value),
                   xytext=(stages[-1] - 0.5, final_value + 0.05),
                   arrowprops=dict(arrowstyle='->', color='black'),
                   fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/progressive_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
