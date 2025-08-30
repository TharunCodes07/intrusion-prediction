"""
Visualization utilities for MSAFN model attention analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from config import Config

class AttentionVisualizer:
    """Visualize attention weights and model interpretability"""
    
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        
    def extract_attention_weights(self, X_sample):
        """Extract attention weights from the model"""
        # Create a model that outputs attention weights
        attention_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=[
                self.model.get_layer('behavioral_stream').attention.dense.output,
                self.model.get_layer('attention_fusion').multi_head_attention.dense.output
            ]
        )
        
        behavioral_attention, fusion_attention = attention_model(X_sample)
        return behavioral_attention, fusion_attention
    
    def plot_attention_heatmap(self, attention_weights, sample_idx=0, save_path=None):
        """Plot attention heatmap for feature importance"""
        plt.figure(figsize=(15, 8))
        
        # Average attention across heads if multi-head
        if len(attention_weights.shape) > 2:
            attention_avg = np.mean(attention_weights[sample_idx], axis=0)
        else:
            attention_avg = attention_weights[sample_idx]
        
        # Create feature importance plot
        plt.subplot(1, 2, 1)
        feature_importance = np.mean(attention_avg, axis=0)
        sorted_idx = np.argsort(feature_importance)[-20:]  # Top 20 features
        
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
        plt.yticks(range(len(sorted_idx)), [self.feature_names[i] for i in sorted_idx])
        plt.xlabel('Attention Weight')
        plt.title('Top 20 Feature Importance (Attention-based)')
        plt.tight_layout()
        
        # Create attention matrix heatmap
        plt.subplot(1, 2, 2)
        sns.heatmap(attention_avg[:10, :20], cmap='viridis', cbar=True)
        plt.xlabel('Feature Index')
        plt.ylabel('Sequence Position')
        plt.title('Attention Weights Heatmap')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_attention_distribution(self, X_samples, y_samples, save_path=None):
        """Plot attention distribution across different attack types"""
        behavioral_attention, fusion_attention = self.extract_attention_weights(X_samples)
        
        # Average attention per class
        unique_classes = np.unique(y_samples)
        fig, axes = plt.subplots(2, len(unique_classes), figsize=(20, 10))
        
        for i, class_label in enumerate(unique_classes):
            class_mask = y_samples == class_label
            class_attention = fusion_attention[class_mask]
            
            if len(class_attention) > 0:
                avg_attention = np.mean(class_attention, axis=0)
                feature_importance = np.mean(avg_attention, axis=0)
                
                # Top features for this class
                top_indices = np.argsort(feature_importance)[-10:]
                
                axes[0, i].bar(range(len(top_indices)), feature_importance[top_indices])
                axes[0, i].set_title(f'Class {class_label} - Top Features')
                axes[0, i].set_xticks(range(len(top_indices)))
                axes[0, i].set_xticklabels([self.feature_names[idx][:10] for idx in top_indices], 
                                         rotation=45, ha='right')
                
                # Attention heatmap for this class
                if len(avg_attention.shape) > 1:
                    sns.heatmap(avg_attention[:5, :20], ax=axes[1, i], cmap='Blues')
                    axes[1, i].set_title(f'Class {class_label} - Attention Pattern')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_stream_contributions(self, X_sample, save_path=None):
        """Visualize contributions from different streams"""
        # Extract outputs from each stream
        temporal_output = self.model.get_layer('temporal_stream')(X_sample)
        statistical_output = self.model.get_layer('statistical_stream')(X_sample)
        behavioral_output, _ = self.model.get_layer('behavioral_stream')(X_sample)
        
        # Calculate contribution magnitudes
        temporal_contrib = np.mean(np.abs(temporal_output), axis=(1, 2))
        statistical_contrib = np.mean(np.abs(statistical_output), axis=(1, 2))
        behavioral_contrib = np.mean(np.abs(behavioral_output), axis=(1, 2))
        
        # Plot contributions
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        streams = ['Temporal', 'Statistical', 'Behavioral']
        contributions = [temporal_contrib, statistical_contrib, behavioral_contrib]
        
        for i, (stream, contrib) in enumerate(zip(streams, contributions)):
            axes[i].hist(contrib, bins=30, alpha=0.7)
            axes[i].set_title(f'{stream} Stream Contributions')
            axes[i].set_xlabel('Contribution Magnitude')
            axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class ModelEvaluator:
    """Comprehensive model evaluation and visualization"""
    
    def __init__(self, attack_names=None):
        if attack_names is None:
            self.attack_names = ['BENIGN', 'Infiltration', 'Brute Force', 'SQL Injection', 'XSS']
        else:
            self.attack_names = attack_names
    
    def plot_training_history(self, history, save_path=None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(history.history['loss'], label='Training Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Accuracy
        axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        
        # Precision
        if 'precision' in history.history:
            axes[1, 0].plot(history.history['precision'], label='Training Precision')
            axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
        
        # Recall
        if 'recall' in history.history:
            axes[1, 1].plot(history.history['recall'], label='Training Recall')
            axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.attack_names, 
                    yticklabels=self.attack_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_classification_report(self, y_true, y_pred):
        """Generate detailed classification report"""
        report = classification_report(y_true, y_pred, 
                                     target_names=self.attack_names,
                                     output_dict=True)
        
        # Convert to DataFrame for better visualization
        df_report = pd.DataFrame(report).transpose()
        
        print("Classification Report:")
        print("=" * 50)
        print(df_report)
        
        return df_report
    
    def plot_class_distribution(self, y_data, title="Class Distribution", save_path=None):
        """Plot class distribution"""
        unique, counts = np.unique(y_data, return_counts=True)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar([self.attack_names[i] for i in unique], counts)
        plt.title(title)
        plt.xlabel('Attack Type')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_models(self, results_dict, save_path=None):
        """Compare multiple model results"""
        metrics = ['accuracy', 'precision', 'recall', 'f1-score']
        models = list(results_dict.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            model_scores = [results_dict[model].get(metric, 0) for model in models]
            
            bars = axes[i].bar(models, model_scores)
            axes[i].set_title(f'{metric.title()} Comparison')
            axes[i].set_ylabel(metric.title())
            axes[i].set_ylim(0, 1)
            
            # Add value labels
            for bar, score in zip(bars, model_scores):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
