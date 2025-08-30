"""
Comparative Analysis between Standard and Progressive MSAFN Training
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# Import experiment classes
from msafn_standard import MSAFNStandardTraining
from msafn_progressive import MSAFNProgressiveTraining
from utils import setup_logging, ExperimentTracker

class MSAFNComparison:
    """
    Compare Standard vs Progressive MSAFN training approaches
    """
    
    def __init__(self, base_config):
        self.base_config = base_config
        self.comparison_dir = 'comparison_results'
        os.makedirs(self.comparison_dir, exist_ok=True)
        
        # Setup logging
        self.logger, _ = setup_logging(self.comparison_dir, 'msafn_comparison')
        
        # Results storage
        self.standard_results = None
        self.progressive_results = None
        
    def run_standard_experiment(self):
        """
        Run standard training experiment
        """
        self.logger.info("Running Standard MSAFN Training...")
        
        standard_config = self.base_config.copy()
        standard_config.update({
            'log_dir': 'logs/msafn_standard',
            'model_dir': 'models/msafn_standard',
            'plots_dir': 'plots/msafn_standard'
        })
        
        experiment = MSAFNStandardTraining(standard_config)
        self.standard_results = experiment.run_complete_experiment()
        
        self.logger.info("Standard training completed")
        return self.standard_results
    
    def run_progressive_experiment(self):
        """
        Run progressive training experiment
        """
        self.logger.info("Running Progressive MSAFN Training...")
        
        progressive_config = self.base_config.copy()
        progressive_config.update({
            'log_dir': 'logs/msafn_progressive',
            'model_dir': 'models/msafn_progressive',
            'plots_dir': 'plots/msafn_progressive',
            'normal_epochs': 50,
            'epochs_per_stage': 30
        })
        
        experiment = MSAFNProgressiveTraining(progressive_config)
        self.progressive_results = experiment.run_complete_experiment()
        
        self.logger.info("Progressive training completed")
        return self.progressive_results
    
    def compare_results(self):
        """
        Compare results between standard and progressive approaches
        """
        if not self.standard_results or not self.progressive_results:
            raise ValueError("Both experiments must be run before comparison")
        
        self.logger.info("Comparing Standard vs Progressive Training Results...")
        
        # Extract results
        standard_metrics = self.standard_results['results']
        progressive_metrics = self.progressive_results['final_results']
        
        # Create comparison DataFrame
        comparison_data = {
            'Metric': ['Accuracy', 'AUC Score', 'Macro F1', 'Weighted F1'],
            'Standard Training': [
                standard_metrics['accuracy'],
                standard_metrics['auc_score'],
                standard_metrics['macro_f1'],
                standard_metrics['weighted_f1']
            ],
            'Progressive Training': [
                progressive_metrics['accuracy'],
                progressive_metrics['auc_score'],
                progressive_metrics['macro_f1'],
                progressive_metrics['weighted_f1']
            ]
        }
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison['Improvement'] = df_comparison['Progressive Training'] - df_comparison['Standard Training']
        df_comparison['Improvement %'] = (df_comparison['Improvement'] / df_comparison['Standard Training']) * 100
        
        # Log comparison
        self.logger.info("\\nComparison Results:")
        self.logger.info(df_comparison.to_string(index=False))
        
        # Save comparison to CSV
        df_comparison.to_csv(f'{self.comparison_dir}/comparison_results.csv', index=False)
        
        return df_comparison
    
    def create_comparison_visualizations(self, df_comparison):
        """
        Create comprehensive comparison visualizations
        """
        self.logger.info("Creating comparison visualizations...")
        
        # 1. Side-by-side bar comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Main comparison chart
        x = np.arange(len(df_comparison))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, df_comparison['Standard Training'], width, 
                       label='Standard Training', color='#FF6B6B', alpha=0.8)
        bars2 = ax1.bar(x + width/2, df_comparison['Progressive Training'], width,
                       label='Progressive Training', color='#4ECDC4', alpha=0.8)
        
        ax1.set_xlabel('Metrics', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Standard vs Progressive Training Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(df_comparison['Metric'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 2. Improvement percentage chart
        colors = ['green' if x > 0 else 'red' for x in df_comparison['Improvement %']]
        bars3 = ax2.bar(df_comparison['Metric'], df_comparison['Improvement %'], 
                       color=colors, alpha=0.7)
        ax2.set_title('Improvement: Progressive vs Standard (%)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Improvement (%)', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels
        for bar, value in zip(bars3, df_comparison['Improvement %']):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height > 0 else -0.3),
                    f'{value:.2f}%', ha='center', va='bottom' if height > 0 else 'top', 
                    fontsize=10, fontweight='bold')
        
        # 3. Training duration comparison
        if 'duration' in self.standard_results and 'duration' in self.progressive_results:
            durations = [
                self.standard_results['duration'].total_seconds() / 3600,  # Convert to hours
                self.progressive_results['duration'].total_seconds() / 3600
            ]
            
            bars4 = ax3.bar(['Standard', 'Progressive'], durations, 
                           color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
            ax3.set_title('Training Duration Comparison', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Duration (hours)', fontsize=12)
            ax3.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, duration in zip(bars4, durations):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{duration:.2f}h', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # 4. Radar chart for comprehensive comparison
        categories = ['Accuracy', 'AUC Score', 'Macro F1', 'Weighted F1']
        standard_values = df_comparison['Standard Training'].values
        progressive_values = df_comparison['Progressive Training'].values
        
        # Number of variables
        N = len(categories)
        
        # Angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Add values for completing the circle
        standard_values = np.concatenate((standard_values, [standard_values[0]]))
        progressive_values = np.concatenate((progressive_values, [progressive_values[0]]))
        
        # Plot
        ax4 = plt.subplot(2, 2, 4, projection='polar')
        ax4.plot(angles, standard_values, 'o-', linewidth=2, label='Standard', color='#FF6B6B')
        ax4.fill(angles, standard_values, alpha=0.25, color='#FF6B6B')
        ax4.plot(angles, progressive_values, 'o-', linewidth=2, label='Progressive', color='#4ECDC4')
        ax4.fill(angles, progressive_values, alpha=0.25, color='#4ECDC4')
        
        # Add category labels
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories)
        ax4.set_ylim(0, 1)
        ax4.set_title('Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
        ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.comparison_dir}/comprehensive_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_progressive_stages_analysis(self):
        """
        Analyze progressive training stages
        """
        if not self.progressive_results or 'stage_results' not in self.progressive_results:
            self.logger.warning("No progressive stage results available for analysis")
            return
        
        stage_results = self.progressive_results['stage_results']
        if not stage_results:
            return
        
        # Extract stage metrics
        stages = list(range(2, len(stage_results) + 2))  # Stages start from 2
        accuracies = [result['accuracy'] for result in stage_results]
        f1_scores = [result['macro_f1'] for result in stage_results]
        auc_scores = [result['auc_score'] for result in stage_results]
        
        # Create stage analysis plot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Accuracy progression
        ax1.plot(stages, accuracies, 'o-', linewidth=3, markersize=10, color='#FF6B6B')
        ax1.set_title('Accuracy by Progressive Stage', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Training Stage', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Add value annotations
        for stage, acc in zip(stages, accuracies):
            ax1.annotate(f'{acc:.3f}', (stage, acc), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=10, fontweight='bold')
        
        # F1-Score progression
        ax2.plot(stages, f1_scores, 's-', linewidth=3, markersize=10, color='#4ECDC4')
        ax2.set_title('F1-Score by Progressive Stage', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Training Stage', fontsize=12)
        ax2.set_ylabel('Macro F1-Score', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Add value annotations
        for stage, f1 in zip(stages, f1_scores):
            ax2.annotate(f'{f1:.3f}', (stage, f1), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=10, fontweight='bold')
        
        # AUC progression
        ax3.plot(stages, auc_scores, '^-', linewidth=3, markersize=10, color='#45B7D1')
        ax3.set_title('AUC by Progressive Stage', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Training Stage', fontsize=12)
        ax3.set_ylabel('AUC Score', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Add value annotations
        for stage, auc in zip(stages, auc_scores):
            ax3.annotate(f'{auc:.3f}', (stage, auc), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.comparison_dir}/progressive_stages_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comprehensive_report(self, df_comparison):
        """
        Generate a comprehensive comparison report
        """
        report_path = f'{self.comparison_dir}/comparison_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# MSAFN Training Approaches Comparison Report\\n\\n")
            f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
            
            f.write("## Executive Summary\\n\\n")
            f.write("This report compares two training approaches for the Multi-Stream Attention-Based ")
            f.write("Fusion Network (MSAFN) on network intrusion detection datasets.\\n\\n")
            
            f.write("### Training Approaches\\n\\n")
            f.write("1. **Standard Training**: Traditional approach training on mixed normal and attack data\\n")
            f.write("2. **Progressive Training**: Sequential approach starting with normal traffic, ")
            f.write("then progressively adding attack types\\n\\n")
            
            f.write("## Results Comparison\\n\\n")
            f.write("| Metric | Standard Training | Progressive Training | Improvement | Improvement % |\\n")
            f.write("|--------|------------------|---------------------|-------------|---------------|\\n")
            
            for _, row in df_comparison.iterrows():
                f.write(f"| {row['Metric']} | {row['Standard Training']:.4f} | ")
                f.write(f"{row['Progressive Training']:.4f} | {row['Improvement']:.4f} | ")
                f.write(f"{row['Improvement %']:.2f}% |\\n")
            
            f.write("\\n## Key Findings\\n\\n")
            
            # Analyze results
            best_approach = "Progressive" if df_comparison['Improvement'].mean() > 0 else "Standard"
            f.write(f"- **Overall Best Approach**: {best_approach} Training\\n")
            
            best_metric = df_comparison.loc[df_comparison['Improvement %'].idxmax(), 'Metric']
            best_improvement = df_comparison['Improvement %'].max()
            f.write(f"- **Largest Improvement**: {best_metric} with {best_improvement:.2f}% improvement\\n")
            
            if 'duration' in self.standard_results and 'duration' in self.progressive_results:
                std_duration = self.standard_results['duration'].total_seconds() / 3600
                prog_duration = self.progressive_results['duration'].total_seconds() / 3600
                f.write(f"- **Training Duration**: Standard {std_duration:.2f}h vs Progressive {prog_duration:.2f}h\\n")
            
            f.write("\\n## Recommendations\\n\\n")
            if df_comparison['Improvement'].mean() > 0:
                f.write("Progressive training shows superior performance and is recommended for:")
                f.write("\\n- Better handling of class imbalance\\n")
                f.write("- Improved attention mechanism learning\\n")
                f.write("- More robust feature representation\\n")
            else:
                f.write("Standard training shows competitive performance and is recommended for:")
                f.write("\\n- Faster training time\\n")
                f.write("- Simpler implementation\\n")
                f.write("- Balanced datasets\\n")
        
        self.logger.info(f"Comprehensive report saved to {report_path}")
    
    def run_complete_comparison(self):
        """
        Run complete comparison between standard and progressive training
        """
        start_time = datetime.now()
        self.logger.info("Starting comprehensive MSAFN comparison study...")
        
        try:
            # Run both experiments
            self.run_standard_experiment()
            self.run_progressive_experiment()
            
            # Compare results
            df_comparison = self.compare_results()
            
            # Create visualizations
            self.create_comparison_visualizations(df_comparison)
            self.create_progressive_stages_analysis()
            
            # Generate report
            self.generate_comprehensive_report(df_comparison)
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.logger.info(f"Complete comparison study finished in {duration}")
            
            return {
                'comparison_df': df_comparison,
                'standard_results': self.standard_results,
                'progressive_results': self.progressive_results,
                'duration': duration
            }
            
        except Exception as e:
            self.logger.error(f"Comparison study failed: {str(e)}")
            raise

def main():
    """
    Main function to run the complete comparison study
    """
    # Set console encoding to UTF-8 for Windows
    import sys
    if sys.platform == "win32":
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
    
    # Base configuration (shared between both approaches)
    base_config = {
        # Data configuration
        'data_files': {
            'infilteration': 'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
            'webattacks': 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv'
        },
        
        # Training configuration
        'epochs': 100,
        'batch_size': 256,
        'learning_rate': 0.001,
        'patience': 15,
        
        # Data split configuration
        'test_size': 0.2,
        'val_size': 0.1,
        'random_state': 42,
        
        # Class imbalance handling
        'handle_imbalance': True,
        'imbalance_method': 'smote_tomek',
        
        # Directory configuration
        'experiment_dir': 'experiments'
    }
    
    # Run comparison
    comparison = MSAFNComparison(base_config)
    results = comparison.run_complete_comparison()
    
    print("\\n" + "="*80)
    print("MSAFN TRAINING APPROACHES COMPARISON COMPLETED!")
    print("="*80)
    print("Results Summary:")
    print(results['comparison_df'].to_string(index=False))
    print(f"\\nTotal Study Duration: {results['duration']}")
    print("\\nDetailed results and visualizations saved in 'comparison_results/' directory")
    print("="*80)

if __name__ == "__main__":
    # Set random seeds for reproducibility
    import numpy as np
    import tensorflow as tf
    
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Run main function
    main()
