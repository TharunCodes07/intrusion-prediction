"""
Multi-Stream Attention-Based Fusion Network (MSAFN) - Standard Training Approach
Network Intrusion Detection System

This implementation uses standard training where the model is trained on
mixed normal and attack data from the beginning.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import logging
from datetime import datetime

# Import custom modules
from data_preprocessing import NetworkDataPreprocessor
from model_components import MSAFNModel, create_callbacks, create_custom_loss, create_metrics
from attention_visualization import AttentionVisualizer
from utils import (setup_logging, save_experiment_config, calculate_class_weights,
                  evaluate_model, plot_training_history, create_experiment_summary, ExperimentTracker)

class MSAFNStandardTraining:
    """
    MSAFN implementation with standard training approach
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.preprocessor = None
        self.attention_visualizer = None
        self.logger = None
        self.experiment_tracker = ExperimentTracker(config['experiment_dir'])
        
        # Setup directories
        os.makedirs(config['log_dir'], exist_ok=True)
        os.makedirs(config['model_dir'], exist_ok=True)
        os.makedirs(config['plots_dir'], exist_ok=True)
        
        # Setup logging
        self.logger, _ = setup_logging(config['log_dir'], 'msafn_standard')
        
        # Save configuration
        config_path = os.path.join(config['log_dir'], 'config.json')
        save_experiment_config(config, config_path)
        
        self.logger.info("MSAFN Standard Training initialized")
        self.logger.info(f"Configuration: {config}")
    
    def load_and_preprocess_data(self):
        """
        Load and preprocess the network intrusion detection datasets
        """
        self.logger.info("Starting data loading and preprocessing...")
        
        # Initialize preprocessor
        self.preprocessor = NetworkDataPreprocessor(log_dir=self.config['log_dir'])
        
        # Load data
        file_paths = [
            self.config['data_files']['infilteration'],
            self.config['data_files']['webattacks']
        ]
        
        combined_df = self.preprocessor.load_data(file_paths)
        
        # Clean data
        combined_df = self.preprocessor.clean_data(combined_df)
        
        # Analyze labels
        label_counts = self.preprocessor.analyze_labels(combined_df)
        
        # Separate features and labels
        X, y = self.preprocessor.separate_features_labels(combined_df)
        
        # Encode labels
        y_encoded = self.preprocessor.encode_labels(y)
        
        # Get feature groups for multi-stream architecture
        self.feature_groups = self.preprocessor.get_feature_groups()
        
        # Prepare datasets for standard training
        datasets = self.preprocessor.prepare_standard_datasets(
            X.values, y_encoded,
            test_size=self.config['test_size'],
            val_size=self.config['val_size'],
            random_state=self.config['random_state']
        )
        
        self.X_train, self.y_train = datasets['train']
        self.X_val, self.y_val = datasets['val']
        self.X_test, self.y_test = datasets['test']
        
        # Handle class imbalance if specified
        if self.config['handle_imbalance']:
            self.logger.info("Handling class imbalance...")
            self.X_train, self.y_train = self.preprocessor.handle_class_imbalance(
                self.X_train, self.y_train,
                method=self.config['imbalance_method'],
                random_state=self.config['random_state']
            )
        
        # Scale features
        self.X_train, self.X_val = self.preprocessor.scale_features(self.X_train, self.X_val)
        self.X_test = self.preprocessor.scaler.transform(self.X_test)
        
        # Calculate class weights
        self.class_weights = calculate_class_weights(self.y_train)
        
        # Initialize attention visualizer
        self.attention_visualizer = AttentionVisualizer(
            feature_names=self.preprocessor.feature_names,
            feature_groups=self.feature_groups,
            save_dir=self.config['plots_dir']
        )
        
        self.logger.info("Data preprocessing completed")
        self.logger.info(f"Training set: {self.X_train.shape}")
        self.logger.info(f"Validation set: {self.X_val.shape}")
        self.logger.info(f"Test set: {self.X_test.shape}")
        self.logger.info(f"Number of classes: {len(np.unique(self.y_train))}")
        self.logger.info(f"Class weights: {self.class_weights}")
    
    def build_model(self):
        """
        Build the MSAFN model
        """
        self.logger.info("Building MSAFN model...")
        
        # Create model
        self.model = MSAFNModel(
            feature_groups=self.feature_groups,
            num_classes=len(np.unique(self.y_train))
        )
        
        # Build model by calling it with sample data
        sample_input = tf.random.normal((1, self.X_train.shape[1]))
        _ = self.model(sample_input)
        
        # Create loss function
        loss_fn = create_custom_loss(class_weights=list(self.class_weights.values()))
        
        # Create metrics
        metrics = create_metrics()
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss=loss_fn,
            metrics=metrics
        )
        
        # Print model summary
        self.model.summary()
        self.logger.info("Model built and compiled successfully")
        
        # Log model architecture to file
        with open(os.path.join(self.config['log_dir'], 'model_summary.txt'), 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    def train_model(self):
        """
        Train the MSAFN model with standard approach
        """
        self.logger.info("Starting model training (Standard Approach)...")
        
        # Create callbacks
        model_save_path = os.path.join(self.config['model_dir'], 'msafn_standard_best.h5')
        tensorboard_dir = os.path.join(self.config['log_dir'], 'tensorboard_standard')
        
        callbacks = create_callbacks(
            model_save_path=model_save_path,
            log_dir=tensorboard_dir,
            patience=self.config['patience']
        )
        
        # Convert labels to categorical if needed
        if len(np.unique(self.y_train)) > 2:
            y_train_cat = tf.keras.utils.to_categorical(self.y_train)
            y_val_cat = tf.keras.utils.to_categorical(self.y_val)
        else:
            y_train_cat = self.y_train
            y_val_cat = self.y_val
        
        # Train the model
        history = self.model.fit(
            self.X_train, y_train_cat,
            validation_data=(self.X_val, y_val_cat),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=1,
            class_weight=self.class_weights
        )
        
        self.logger.info("Training completed")
        
        # Plot training history
        plot_training_history(history, self.config['plots_dir'])
        
        # Load best model
        self.model = tf.keras.models.load_model(model_save_path, compile=False)
        
        # Recompile with metrics for evaluation
        loss_fn = create_custom_loss(class_weights=list(self.class_weights.values()))
        metrics = create_metrics()
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss=loss_fn,
            metrics=metrics
        )
        
        return history
    
    def evaluate_model(self):
        """
        Comprehensive model evaluation
        """
        self.logger.info("Starting model evaluation...")
        
        # Get class names
        label_encoder = self.preprocessor.label_encoder
        class_names = label_encoder.classes_
        
        # Evaluate model
        results = evaluate_model(
            model=self.model,
            X_test=self.X_test,
            y_test=self.y_test,
            class_names=class_names,
            save_dir=self.config['plots_dir']
        )
        
        # Log results
        self.logger.info("Evaluation Results:")
        self.logger.info(f"Accuracy: {results['accuracy']:.4f}")
        self.logger.info(f"AUC Score: {results['auc_score']:.4f}")
        self.logger.info(f"Macro F1: {results['macro_f1']:.4f}")
        self.logger.info(f"Weighted F1: {results['weighted_f1']:.4f}")
        
        # Save detailed results
        summary = create_experiment_summary(
            results=results,
            config=self.config,
            save_path=os.path.join(self.config['log_dir'], 'experiment_summary.json')
        )
        
        # Add to experiment tracker
        self.experiment_tracker.add_experiment(
            name='MSAFN_Standard',
            results=results,
            config=self.config
        )
        
        return results
    
    def visualize_attention(self):
        """
        Create attention visualizations
        """
        self.logger.info("Creating attention visualizations...")
        
        # Create comprehensive attention dashboard
        self.attention_visualizer.create_attention_dashboard(
            model=self.model,
            X_test=self.X_test,
            y_test=self.y_test
        )
        
        self.logger.info("Attention visualizations completed")
    
    def run_complete_experiment(self):
        """
        Run the complete MSAFN standard training experiment
        """
        start_time = datetime.now()
        self.logger.info(f"Starting complete MSAFN Standard Training experiment at {start_time}")
        
        try:
            # Step 1: Load and preprocess data
            self.load_and_preprocess_data()
            
            # Step 2: Build model
            self.build_model()
            
            # Step 3: Train model
            history = self.train_model()
            
            # Step 4: Evaluate model
            results = self.evaluate_model()
            
            # Step 5: Visualize attention
            self.visualize_attention()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.logger.info(f"Experiment completed successfully in {duration}")
            self.logger.info(f"Final Results Summary:")
            self.logger.info(f"  - Accuracy: {results['accuracy']:.4f}")
            self.logger.info(f"  - AUC Score: {results['auc_score']:.4f}")
            self.logger.info(f"  - Macro F1: {results['macro_f1']:.4f}")
            
            return {
                'model': self.model,
                'history': history,
                'results': results,
                'duration': duration
            }
            
        except Exception as e:
            self.logger.error(f"Experiment failed with error: {str(e)}")
            raise

def main():
    """
    Main function to run MSAFN standard training
    """
    # Set console encoding to UTF-8 for Windows
    import sys
    if sys.platform == "win32":
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
    
    # Configuration
    config = {
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
        'log_dir': 'logs/msafn_standard',
        'model_dir': 'models/msafn_standard',
        'plots_dir': 'plots/msafn_standard',
        'experiment_dir': 'experiments'
    }
    
    # Create and run experiment
    experiment = MSAFNStandardTraining(config)
    results = experiment.run_complete_experiment()
    
    print("\\n" + "="*60)
    print("MSAFN STANDARD TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Final Accuracy: {results['results']['accuracy']:.4f}")
    print(f"Final AUC Score: {results['results']['auc_score']:.4f}")
    print(f"Final Macro F1: {results['results']['macro_f1']:.4f}")
    print(f"Training Duration: {results['duration']}")
    print("="*60)

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Run main function
    main()
