import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tensorflow as tf
import pandas as pd
import os

class AttentionVisualizer:
    """
    Comprehensive attention visualization for MSAFN model
    """
    
    def __init__(self, feature_names, feature_groups, save_dir='attention_plots'):
        self.feature_names = feature_names
        self.feature_groups = feature_groups
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Create feature group names mapping
        self.group_names = ['Temporal', 'Statistical', 'Behavioral']
    
    def plot_stream_attention_weights(self, attention_weights, epoch=None, save=True):
        """
        Plot attention weights for different streams
        """
        # Convert to numpy if tensor
        if isinstance(attention_weights, tf.Tensor):
            attention_weights = attention_weights.numpy()
        
        # Average across batch dimension
        avg_weights = np.mean(attention_weights, axis=0)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        bars = ax1.bar(self.group_names, avg_weights, 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
        ax1.set_title('Stream Attention Weights', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Attention Weight', fontsize=12)
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, weight in zip(bars, avg_weights):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{weight:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        wedges, texts, autotexts = ax2.pie(avg_weights, labels=self.group_names, 
                                          autopct='%1.1f%%', colors=colors,
                                          startangle=90, textprops={'fontsize': 10})
        ax2.set_title('Stream Attention Distribution', fontsize=16, fontweight='bold')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        
        if save:
            epoch_str = f'_epoch_{epoch}' if epoch is not None else ''
            plt.savefig(f'{self.save_dir}/stream_attention{epoch_str}.png', 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
        return avg_weights
    
    def plot_interactive_attention_heatmap(self, model, X_sample, y_sample, sample_indices=None):
        """
        Create interactive attention heatmap using Plotly
        """
        if sample_indices is None:
            sample_indices = np.random.choice(len(X_sample), min(100, len(X_sample)), replace=False)
        
        X_subset = X_sample[sample_indices]
        y_subset = y_sample[sample_indices]
        
        # Get predictions and attention weights
        predictions = model(X_subset, training=False)
        attention_weights = model.get_attention_weights()
        
        if isinstance(attention_weights, tf.Tensor):
            attention_weights = attention_weights.numpy()
        
        # Create heatmap data
        heatmap_data = attention_weights.squeeze()
        
        # Create interactive heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=self.group_names,
            y=[f'Sample {i}' for i in sample_indices],
            colorscale='RdYlBu_r',
            colorbar=dict(title='Attention Weight'),
            hoverongaps=False,
            hovertemplate='Stream: %{x}<br>Sample: %{y}<br>Weight: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Attention Weights Heatmap Across Samples',
            xaxis_title='Feature Streams',
            yaxis_title='Samples',
            font=dict(size=12),
            height=600
        )
        
        # Save interactive plot
        fig.write_html(f'{self.save_dir}/interactive_attention_heatmap.html')
        fig.show()
        
        return fig
    
    def plot_feature_importance_by_stream(self, model, X_sample, top_k=10):
        """
        Plot feature importance within each stream
        """
        # Get a sample batch
        sample_batch = X_sample[:min(1000, len(X_sample))]
        
        # Extract features for each stream
        temporal_features = tf.gather(sample_batch, self.feature_groups['temporal'], axis=1)
        statistical_features = tf.gather(sample_batch, self.feature_groups['statistical'], axis=1)
        behavioral_features = tf.gather(sample_batch, self.feature_groups['behavioral'], axis=1)
        
        # Calculate feature importance using gradients
        with tf.GradientTape() as tape:
            tape.watch(sample_batch)
            predictions = model(sample_batch, training=False)
            
        gradients = tape.gradient(predictions, sample_batch)
        
        if gradients is not None:
            # Calculate importance as mean absolute gradient
            importance = tf.reduce_mean(tf.abs(gradients), axis=0).numpy()
            
            # Create subplots for each stream
            fig, axes = plt.subplots(3, 1, figsize=(15, 18))
            
            streams = ['temporal', 'statistical', 'behavioral']
            stream_titles = ['Temporal Features', 'Statistical Features', 'Behavioral Features']
            
            for i, (stream, title) in enumerate(zip(streams, stream_titles)):
                indices = self.feature_groups[stream]
                stream_importance = importance[indices]
                stream_features = [self.feature_names[idx] for idx in indices]
                
                # Sort by importance
                sorted_indices = np.argsort(stream_importance)[::-1][:top_k]
                top_importance = stream_importance[sorted_indices]
                top_features = [stream_features[idx] for idx in sorted_indices]
                
                # Create horizontal bar plot
                y_pos = np.arange(len(top_features))
                bars = axes[i].barh(y_pos, top_importance, 
                                   color=plt.cm.Set3(np.linspace(0, 1, len(top_features))))
                
                axes[i].set_yticks(y_pos)
                axes[i].set_yticklabels(top_features, fontsize=10)
                axes[i].set_xlabel('Feature Importance (Mean |Gradient|)', fontsize=12)
                axes[i].set_title(f'{title} - Top {top_k} Important Features', 
                                 fontsize=14, fontweight='bold')
                axes[i].grid(axis='x', alpha=0.3)
                
                # Add value labels
                for j, (bar, imp) in enumerate(zip(bars, top_importance)):
                    width = bar.get_width()
                    axes[i].text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                               f'{imp:.4f}', ha='left', va='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(f'{self.save_dir}/feature_importance_by_stream.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
    
    def plot_attention_evolution(self, attention_history, save=True):
        """
        Plot how attention weights evolve during training
        """
        if not attention_history:
            print("No attention history available")
            return
        
        # Convert to numpy arrays
        epochs = list(range(len(attention_history)))
        temporal_weights = [weights[0] for weights in attention_history]
        statistical_weights = [weights[1] for weights in attention_history]
        behavioral_weights = [weights[2] for weights in attention_history]
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        plt.plot(epochs, temporal_weights, 'o-', label='Temporal Stream', 
                linewidth=2, markersize=6, color='#FF6B6B')
        plt.plot(epochs, statistical_weights, 's-', label='Statistical Stream', 
                linewidth=2, markersize=6, color='#4ECDC4')
        plt.plot(epochs, behavioral_weights, '^-', label='Behavioral Stream', 
                linewidth=2, markersize=6, color='#45B7D1')
        
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Attention Weight', fontsize=14)
        plt.title('Evolution of Stream Attention Weights During Training', 
                 fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, max(epochs))
        plt.ylim(0, 1)
        
        # Add annotations for final weights
        final_temporal = temporal_weights[-1]
        final_statistical = statistical_weights[-1]
        final_behavioral = behavioral_weights[-1]
        
        plt.annotate(f'Final: {final_temporal:.3f}', 
                    xy=(epochs[-1], final_temporal), 
                    xytext=(epochs[-1]-len(epochs)*0.1, final_temporal+0.05),
                    arrowprops=dict(arrowstyle='->', color='#FF6B6B'),
                    fontsize=10, color='#FF6B6B')
        
        plt.annotate(f'Final: {final_statistical:.3f}', 
                    xy=(epochs[-1], final_statistical), 
                    xytext=(epochs[-1]-len(epochs)*0.1, final_statistical+0.05),
                    arrowprops=dict(arrowstyle='->', color='#4ECDC4'),
                    fontsize=10, color='#4ECDC4')
        
        plt.annotate(f'Final: {final_behavioral:.3f}', 
                    xy=(epochs[-1], final_behavioral), 
                    xytext=(epochs[-1]-len(epochs)*0.1, final_behavioral+0.05),
                    arrowprops=dict(arrowstyle='->', color='#45B7D1'),
                    fontsize=10, color='#45B7D1')
        
        if save:
            plt.savefig(f'{self.save_dir}/attention_evolution.png', 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_attention_dashboard(self, model, X_test, y_test, attention_history=None):
        """
        Create a comprehensive attention analysis dashboard
        """
        print("Creating Attention Analysis Dashboard...")
        
        # 1. Stream attention weights
        predictions = model(X_test[:100], training=False)
        attention_weights = model.get_attention_weights()
        self.plot_stream_attention_weights(attention_weights)
        
        # 2. Interactive heatmap
        self.plot_interactive_attention_heatmap(model, X_test, y_test)
        
        # 3. Feature importance by stream
        self.plot_feature_importance_by_stream(model, X_test)
        
        # 4. Attention evolution (if history available)
        if attention_history:
            self.plot_attention_evolution(attention_history)
        
        # 5. Class-specific attention analysis
        self.plot_class_specific_attention(model, X_test, y_test)
        
        print(f"Dashboard saved to {self.save_dir}/")
    
    def plot_class_specific_attention(self, model, X_test, y_test, max_samples_per_class=50):
        """
        Analyze attention patterns for different classes
        """
        unique_classes = np.unique(y_test)
        class_attention_data = []
        
        for class_label in unique_classes:
            # Get samples for this class
            class_indices = np.where(y_test == class_label)[0]
            if len(class_indices) > max_samples_per_class:
                class_indices = np.random.choice(class_indices, max_samples_per_class, replace=False)
            
            if len(class_indices) > 0:
                class_samples = X_test[class_indices]
                
                # Get attention weights for this class
                predictions = model(class_samples, training=False)
                attention_weights = model.get_attention_weights()
                
                if isinstance(attention_weights, tf.Tensor):
                    attention_weights = attention_weights.numpy()
                
                # Average attention weights for this class
                avg_attention = np.mean(attention_weights, axis=0)
                
                for i, stream_name in enumerate(self.group_names):
                    class_attention_data.append({
                        'Class': f'Class {class_label}',
                        'Stream': stream_name,
                        'Attention_Weight': avg_attention[i]
                    })
        
        # Create DataFrame
        df_attention = pd.DataFrame(class_attention_data)
        
        # Create grouped bar plot
        plt.figure(figsize=(12, 8))
        
        # Pivot for easier plotting
        pivot_df = df_attention.pivot(index='Class', columns='Stream', values='Attention_Weight')
        
        ax = pivot_df.plot(kind='bar', width=0.8, figsize=(12, 8), 
                          color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        
        plt.title('Class-Specific Attention Patterns', fontsize=16, fontweight='bold')
        plt.xlabel('Class', fontsize=14)
        plt.ylabel('Average Attention Weight', fontsize=14)
        plt.legend(title='Feature Stream', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', rotation=90, fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/class_specific_attention.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return df_attention
