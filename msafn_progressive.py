"""
Multi-Stream Attention-Based Fusion Network (MSAFN) - Progressive Training Approach
Network Intrusion Detection System

This implementation uses progressive training where the model is first trained
on normal traffic, then progressively adds different attack types.
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
                  evaluate_model, plot_training_history, create_experiment_summary, 
                  ExperimentTracker, create_progressive_training_visualization)

class MSAFNProgressiveTraining:
    """
    MSAFN implementation with progressive training approach
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.preprocessor = None
        self.attention_visualizer = None
        self.logger = None
        self.experiment_tracker = ExperimentTracker(config['experiment_dir'])
        self.stage_results = []
        self.attention_history = []
        
        # Setup directories
        os.makedirs(config['log_dir'], exist_ok=True)
        os.makedirs(config['model_dir'], exist_ok=True)
        os.makedirs(config['plots_dir'], exist_ok=True)
        
        # Setup logging
        self.logger, _ = setup_logging(config['log_dir'], 'msafn_progressive')
        
        # Save configuration
        config_path = os.path.join(config['log_dir'], 'config.json')
        save_experiment_config(config, config_path)
        
        self.logger.info("MSAFN Progressive Training initialized")
        self.logger.info(f"Configuration: {config}")
    
    def load_and_preprocess_data(self):
        """
        Load and preprocess data for progressive training
        """
        self.logger.info("Starting data loading and preprocessing for progressive training...")
        
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
        
        # Create progressive datasets
        self.progressive_data = self.preprocessor.create_progressive_datasets(
            X.values, y_encoded,
            test_size=self.config['test_size'],
            val_size=self.config['val_size'],
            random_state=self.config['random_state']
        )
        
        # Scale features for all datasets
        # Scale normal data
        X_normal_train, y_normal_train = self.progressive_data['normal_train']
        X_normal_val, y_normal_val = self.progressive_data['normal_val']
        X_test, y_test = self.progressive_data['test']
        
        X_normal_train_scaled, X_normal_val_scaled = self.preprocessor.scale_features(
            X_normal_train, X_normal_val
        )
        X_test_scaled = self.preprocessor.scaler.transform(X_test)
        
        # Update progressive data with scaled features
        self.progressive_data['normal_train'] = (X_normal_train_scaled, y_normal_train)
        self.progressive_data['normal_val'] = (X_normal_val_scaled, y_normal_val)
        self.progressive_data['test'] = (X_test_scaled, y_test)
        
        # Scale incremental datasets
        for i, dataset in enumerate(self.progressive_data['incremental_datasets']):
            X_inc_train_scaled = self.preprocessor.scaler.transform(dataset['X_train'])
            X_inc_val_scaled = self.preprocessor.scaler.transform(dataset['X_val'])
            
            self.progressive_data['incremental_datasets'][i]['X_train'] = X_inc_train_scaled
            self.progressive_data['incremental_datasets'][i]['X_val'] = X_inc_val_scaled
        
        # Initialize attention visualizer
        self.attention_visualizer = AttentionVisualizer(
            feature_names=self.preprocessor.feature_names,
            feature_groups=self.feature_groups,
            save_dir=self.config['plots_dir']
        )
        
        self.logger.info("Progressive data preprocessing completed")
        self.logger.info(f"Normal training set: {X_normal_train_scaled.shape}")
        self.logger.info(f"Normal validation set: {X_normal_val_scaled.shape}")
        self.logger.info(f"Test set: {X_test_scaled.shape}")
        self.logger.info(f"Number of incremental stages: {len(self.progressive_data['incremental_datasets'])}")
    
    def build_model(self):
        """
        Build the MSAFN model
        """
        self.logger.info("Building MSAFN model for progressive training...")
        
        # Get the number of classes from the final incremental dataset
        final_dataset = self.progressive_data['incremental_datasets'][-1]
        num_classes = len(np.unique(final_dataset['y_train']))
        
        # Create model
        self.model = MSAFNModel(
            feature_groups=self.feature_groups,
            num_classes=num_classes
        )
        
        # Build model by calling it with sample data
        X_normal_train, _ = self.progressive_data['normal_train']
        sample_input = tf.random.normal((1, X_normal_train.shape[1]))
        _ = self.model(sample_input)
        
        # Print model summary
        self.model.summary()
        self.logger.info("Model built successfully")
        
        # Log model architecture to file
        with open(os.path.join(self.config['log_dir'], 'model_summary.txt'), 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    def train_stage_1_normal_only(self):
        """
        Stage 1: Train on normal traffic only
        """
        self.logger.info("="*60)
        self.logger.info("STAGE 1: Training on Normal Traffic Only")
        self.logger.info("="*60)
        
        # Get normal data
        X_train, y_train = self.progressive_data['normal_train']
        X_val, y_val = self.progressive_data['normal_val']
        
        # For normal-only training, we use binary classification (normal=1, anomaly=0)
        # But since we only have normal data, we'll use an autoencoder-like approach
        # or modify to detect anomalies later
        
        # Create loss function and metrics for normal training
        loss_fn = 'mse'  # Use MSE for reconstruction-like training
        metrics = ['mae']
        
        # Compile model for normal training (reconstruction task)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss=loss_fn,
            metrics=metrics
        )
        
        # Create callbacks
        model_save_path = os.path.join(self.config['model_dir'], 'msafn_progressive_stage1.h5')
        tensorboard_dir = os.path.join(self.config['log_dir'], 'tensorboard_stage1')
        
        callbacks = create_callbacks(
            model_save_path=model_save_path,
            log_dir=tensorboard_dir,
            patience=self.config['patience']
        )
        
        # For normal-only training, use the input as target (autoencoder approach)
        history = self.model.fit(
            X_train, X_train,  # Input as target for reconstruction
            validation_data=(X_val, X_val),
            epochs=self.config['normal_epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Store attention weights
        sample_batch = X_train[:32]
        _ = self.model(sample_batch, training=False)
        attention_weights = self.model.get_attention_weights()
        if attention_weights is not None:
            avg_attention = np.mean(attention_weights.numpy(), axis=0)
            self.attention_history.append(avg_attention)
        
        self.logger.info("Stage 1 (Normal-only) training completed")
        
        # Visualize attention weights for this stage
        self.attention_visualizer.plot_stream_attention_weights(
            attention_weights, epoch=f"stage_1", save=True
        )
        
        return history
    
    def train_incremental_stages(self):
        """
        Progressive training stages with incremental attack types
        """
        X_test, y_test = self.progressive_data['test']
        
        for stage_idx, dataset in enumerate(self.progressive_data['incremental_datasets']):
            stage_num = stage_idx + 2  # Stage 2, 3, 4, etc.
            
            self.logger.info("="*60)
            self.logger.info(f"STAGE {stage_num}: Adding Attack Types {dataset['attack_types']}")
            self.logger.info("="*60)
            
            # Get data for this stage
            X_train = dataset['X_train']
            y_train = dataset['y_train']
            X_val = dataset['X_val']
            y_val = dataset['y_val']
            
            # Calculate class weights for this stage
            class_weights = calculate_class_weights(y_train)
            
            # Handle class imbalance if specified
            if self.config['handle_imbalance'] and stage_num > 2:  # Skip for first attack stage if too few samples
                try:
                    X_train, y_train = self.preprocessor.handle_class_imbalance(
                        X_train, y_train,
                        method=self.config['imbalance_method'],
                        random_state=self.config['random_state']
                    )
                    # Recalculate class weights after resampling
                    class_weights = calculate_class_weights(y_train)
                except Exception as e:
                    self.logger.warning(f"Could not apply SMOTE in stage {stage_num}: {str(e)}")
            
            # Create loss function and metrics for classification
            num_classes = len(np.unique(y_train))
            loss_fn = create_custom_loss(class_weights=list(class_weights.values()))
            metrics = create_metrics()
            
            # Recompile model for classification
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
                loss=loss_fn,
                metrics=metrics
            )
            
            # Create callbacks for this stage
            model_save_path = os.path.join(self.config['model_dir'], f'msafn_progressive_stage{stage_num}.h5')
            tensorboard_dir = os.path.join(self.config['log_dir'], f'tensorboard_stage{stage_num}')
            
            callbacks = create_callbacks(
                model_save_path=model_save_path,
                log_dir=tensorboard_dir,
                patience=self.config['patience']
            )
            
            # Convert labels to categorical
            y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
            y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)
            
            # Train for this stage
            history = self.model.fit(
                X_train, y_train_cat,
                validation_data=(X_val, y_val_cat),
                epochs=self.config['epochs_per_stage'],
                batch_size=self.config['batch_size'],
                callbacks=callbacks,
                verbose=1,
                class_weight=class_weights
            )
            
            # Load best model for this stage
            self.model = tf.keras.models.load_model(model_save_path, compile=False)
            
            # Recompile for evaluation
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
                loss=loss_fn,
                metrics=metrics
            )
            
            # Evaluate on test set
            self.logger.info(f"Evaluating Stage {stage_num}...")
            
            # For evaluation, we need to ensure test labels match the current number of classes
            # Map test labels to current stage classes
            unique_train_classes = np.unique(y_train)
            test_mask = np.isin(y_test, unique_train_classes)
            
            if np.sum(test_mask) > 0:
                X_test_stage = X_test[test_mask]
                y_test_stage = y_test[test_mask]
                
                # Get class names for this stage
                label_encoder = self.preprocessor.label_encoder
                class_names = [label_encoder.classes_[i] for i in unique_train_classes]
                
                # Evaluate
                results = evaluate_model(
                    model=self.model,
                    X_test=X_test_stage,
                    y_test=y_test_stage,
                    class_names=class_names,
                    save_dir=os.path.join(self.config['plots_dir'], f'stage_{stage_num}')
                )
                
                self.stage_results.append(results)
                
                self.logger.info(f"Stage {stage_num} Results:")
                self.logger.info(f"  Accuracy: {results['accuracy']:.4f}")
                self.logger.info(f"  AUC Score: {results['auc_score']:.4f}")
                self.logger.info(f"  Macro F1: {results['macro_f1']:.4f}")
            
            # Store attention weights for this stage
            sample_batch = X_train[:32]
            _ = self.model(sample_batch, training=False)
            attention_weights = self.model.get_attention_weights()
            if attention_weights is not None:
                avg_attention = np.mean(attention_weights.numpy(), axis=0)
                self.attention_history.append(avg_attention)
            
            # Visualize attention weights for this stage
            self.attention_visualizer.plot_stream_attention_weights(
                attention_weights, epoch=f"stage_{stage_num}", save=True
            )
            
            self.logger.info(f"Stage {stage_num} completed")
    
    def final_evaluation(self):
        """
        Final comprehensive evaluation on complete test set
        """
        self.logger.info("="*60)
        self.logger.info("FINAL EVALUATION ON COMPLETE TEST SET")
        self.logger.info("="*60)
        
        X_test, y_test = self.progressive_data['test']
        
        # Get class names
        label_encoder = self.preprocessor.label_encoder
        class_names = label_encoder.classes_
        
        # Final evaluation
        final_results = evaluate_model(
            model=self.model,
            X_test=X_test,
            y_test=y_test,
            class_names=class_names,
            save_dir=self.config['plots_dir']
        )
        
        # Log final results
        self.logger.info("Final Progressive Training Results:")
        self.logger.info(f"Accuracy: {final_results['accuracy']:.4f}")
        self.logger.info(f"AUC Score: {final_results['auc_score']:.4f}")
        self.logger.info(f"Macro F1: {final_results['macro_f1']:.4f}")
        self.logger.info(f"Weighted F1: {final_results['weighted_f1']:.4f}")
        
        # Save detailed results
        summary = create_experiment_summary(
            results=final_results,
            config=self.config,
            save_path=os.path.join(self.config['log_dir'], 'final_experiment_summary.json')
        )
        
        # Add to experiment tracker
        self.experiment_tracker.add_experiment(
            name='MSAFN_Progressive',
            results=final_results,
            config=self.config
        )
        
        return final_results
    
    def visualize_progressive_results(self):
        """
        Create visualizations for progressive training results
        """
        self.logger.info("Creating progressive training visualizations...")
        
        # Plot progressive training results
        if self.stage_results:
            create_progressive_training_visualization(
                self.stage_results, 
                self.config['plots_dir']
            )
        
        # Plot attention evolution
        if self.attention_history:
            self.attention_visualizer.plot_attention_evolution(
                self.attention_history, 
                save=True
            )
        
        # Create comprehensive attention dashboard
        X_test, y_test = self.progressive_data['test']
        self.attention_visualizer.create_attention_dashboard(
            model=self.model,
            X_test=X_test,
            y_test=y_test,
            attention_history=self.attention_history
        )
        
        self.logger.info("Progressive training visualizations completed")
    
    def run_complete_experiment(self):
        """
        Run the complete MSAFN progressive training experiment
        """
        start_time = datetime.now()
        self.logger.info(f"Starting complete MSAFN Progressive Training experiment at {start_time}")
        
        try:
            # Step 1: Load and preprocess data
            self.load_and_preprocess_data()
            
            # Step 2: Build model
            self.build_model()
            
            # Step 3: Train Stage 1 (Normal only)
            stage1_history = self.train_stage_1_normal_only()
            
            # Step 4: Train incremental stages
            self.train_incremental_stages()
            
            # Step 5: Final evaluation
            final_results = self.final_evaluation()
            
            # Step 6: Create visualizations
            self.visualize_progressive_results()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.logger.info(f"Progressive training experiment completed successfully in {duration}")
            self.logger.info(f"Final Results Summary:")
            self.logger.info(f"  - Accuracy: {final_results['accuracy']:.4f}")
            self.logger.info(f"  - AUC Score: {final_results['auc_score']:.4f}")
            self.logger.info(f"  - Macro F1: {final_results['macro_f1']:.4f}")
            
            return {
                'model': self.model,
                'stage_results': self.stage_results,
                'final_results': final_results,
                'attention_history': self.attention_history,
                'duration': duration
            }
            
        except Exception as e:
            self.logger.error(f"Progressive training experiment failed with error: {str(e)}")
            raise

def main():
    """
    Main function to run MSAFN progressive training
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
        'normal_epochs': 50,  # Epochs for normal-only training
        'epochs_per_stage': 30,  # Epochs per incremental stage
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
        'log_dir': 'logs/msafn_progressive',
        'model_dir': 'models/msafn_progressive',
        'plots_dir': 'plots/msafn_progressive',
        'experiment_dir': 'experiments'
    }
    
    # Create and run experiment
    experiment = MSAFNProgressiveTraining(config)
    results = experiment.run_complete_experiment()
    
    print("\\n" + "="*60)
    print("MSAFN PROGRESSIVE TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Final Accuracy: {results['final_results']['accuracy']:.4f}")
    print(f"Final AUC Score: {results['final_results']['auc_score']:.4f}")
    print(f"Final Macro F1: {results['final_results']['macro_f1']:.4f}")
    print(f"Number of Progressive Stages: {len(results['stage_results']) + 1}")
    print(f"Training Duration: {results['duration']}")
    print("="*60)

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Run main function
    main()
