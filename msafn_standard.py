"""
MSAFN Model Training - Standard Approach
Multi-Stream Attention-Based Fusion Network for Network Intrusion Detection
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import os
import datetime

# GPU optimization settings
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        print(f"GPU acceleration enabled. Found {len(gpus)} GPU(s)")
        print("Memory growth enabled for efficient GPU utilization")
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
else:
    print("No GPUs found, using CPU")

# Import custom modules
from config import Config
from data_preprocessor import DataPreprocessor
from msafn_components import build_msafn_model, AdversarialTraining
from visualization import AttentionVisualizer, ModelEvaluator

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class MSAFNTrainer:
    """Standard MSAFN trainer without progressive training"""
    
    def __init__(self):
        self.model = None
        self.history = None
        self.preprocessor = DataPreprocessor()
        self.evaluator = ModelEvaluator()
        
    def prepare_data(self):
        """Prepare and split data for training"""
        print("=" * 60)
        print("PREPARING DATA FOR STANDARD TRAINING")
        print("=" * 60)
        
        # Load and preprocess data
        X, y = self.preprocessor.prepare_data(apply_balancing=True)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.split_data(X, y)
        
        # Convert to categorical
        y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=Config.NUM_CLASSES)
        y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=Config.NUM_CLASSES)
        y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=Config.NUM_CLASSES)
        
        return (X_train, X_val, X_test), (y_train_cat, y_val_cat, y_test_cat), (y_train, y_val, y_test)
    
    def build_model(self):
        """Build and compile the MSAFN model"""
        print("\\nBuilding MSAFN Model...")
        
        self.model = build_msafn_model()
        
        # Print model summary
        self.model.summary()
        
        # Compile model with advanced metrics
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        return self.model
    
    def setup_callbacks(self):
        """Setup training callbacks with best practices:
        - EarlyStopping: Stops training when val_loss stops improving
        - ReduceLROnPlateau: Reduces learning rate when val_loss plateaus
        - ModelCheckpoint: Saves best model based on val_accuracy
        """
        callbacks = []
        
        # Early stopping - improved configuration
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,  # Reduced patience for faster stopping
            min_delta=0.001,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Model checkpoint
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(Config.MODEL_SAVE_PATH, 'msafn_standard_best.keras'),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(model_checkpoint)
        
        # Reduce learning rate on plateau - improved configuration
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,  # Reduce LR by 80%
            patience=5,  # Reduce patience for more responsive LR reduction
            min_lr=0.001,  # Higher minimum LR
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # TensorBoard logging
        log_dir = os.path.join(Config.LOGS_PATH, f"msafn_standard_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        callbacks.append(tensorboard)
        
        return callbacks
    
    def create_optimized_dataset(self, X, y, batch_size, shuffle=True):
        """Create optimized tf.data.Dataset for better GPU utilization"""
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        if shuffle:
            # Shuffle with a large buffer for good randomization
            dataset = dataset.shuffle(buffer_size=10000)
        
        # Batch the data
        dataset = dataset.batch(batch_size)
        
        # Prefetch for performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def train_model(self, X_data, y_data):
        """Train the model with standard approach"""
        X_train, X_val, X_test = X_data
        y_train, y_val, y_test = y_data
        
        print("\\n" + "=" * 60)
        print("STARTING STANDARD TRAINING")
        print("=" * 60)
        
        # Create optimized datasets
        train_dataset = self.create_optimized_dataset(X_train, y_train, Config.BATCH_SIZE, shuffle=True)
        val_dataset = self.create_optimized_dataset(X_val, y_val, Config.BATCH_SIZE, shuffle=False)
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Train model with optimized datasets
        self.history = self.model.fit(
            train_dataset,
            epochs=Config.EPOCHS,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, X_test, y_test, y_test_original):
        """Comprehensive model evaluation"""
        print("\\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        
        # Predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Evaluation metrics
        test_loss, test_accuracy, test_precision, test_recall, test_f1 = self.model.evaluate(
            X_test, y_test, verbose=0
        )
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test F1-Score: {test_f1:.4f}")
        
        # Classification report
        report = self.evaluator.generate_classification_report(y_test_original, y_pred)
        
        # Confusion matrix
        self.evaluator.plot_confusion_matrix(
            y_test_original, y_pred,
            save_path=os.path.join(Config.PLOTS_PATH, 'confusion_matrix_standard.png')
        )
        
        # Training history
        self.evaluator.plot_training_history(
            self.history,
            save_path=os.path.join(Config.PLOTS_PATH, 'training_history_standard.png')
        )
        
        return {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': test_f1,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def visualize_attention(self, X_sample, y_sample):
        """Visualize attention weights"""
        print("\\nGenerating Attention Visualizations...")
        
        # Initialize visualizer
        visualizer = AttentionVisualizer(self.model, self.preprocessor.feature_names)
        
        # Plot attention heatmap
        try:
            behavioral_attention, fusion_attention = visualizer.extract_attention_weights(X_sample[:100])
            
            visualizer.plot_attention_heatmap(
                fusion_attention,
                save_path=os.path.join(Config.PLOTS_PATH, 'attention_heatmap_standard.png')
            )
            
            visualizer.plot_feature_attention_distribution(
                X_sample[:100], y_sample[:100],
                save_path=os.path.join(Config.PLOTS_PATH, 'feature_attention_standard.png')
            )
            
            visualizer.plot_stream_contributions(
                X_sample[:50],
                save_path=os.path.join(Config.PLOTS_PATH, 'stream_contributions_standard.png')
            )
            
        except Exception as e:
            print(f"Warning: Could not generate attention visualizations: {e}")
    
    def save_model(self):
        """Save the trained model"""
        model_path = os.path.join(Config.MODEL_SAVE_PATH, 'msafn_standard_final.keras')
        self.model.save(model_path)
        print(f"\\nModel saved to: {model_path}")
        
        # Save model architecture as JSON
        model_json = self.model.to_json()
        with open(os.path.join(Config.MODEL_SAVE_PATH, 'msafn_standard_architecture.json'), 'w') as json_file:
            json_file.write(model_json)
        
        return model_path

def main():
    """Main training pipeline"""
    print("MSAFN Network Intrusion Detection - Standard Training")
    print("=" * 60)
    
    # Initialize trainer
    trainer = MSAFNTrainer()
    
    try:
        # Prepare data
        X_data, y_data_cat, y_data_original = trainer.prepare_data()
        X_train, X_val, X_test = X_data
        
        # Build model
        model = trainer.build_model()
        
        # Train model
        history = trainer.train_model(X_data, y_data_cat)
        
        # Evaluate model
        results = trainer.evaluate_model(X_test, y_data_cat[2], y_data_original[2])
        
        # Visualize attention
        trainer.visualize_attention(X_test, y_data_original[2])
        
        # Save model
        model_path = trainer.save_model()
        
        print("\\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Final Results:")
        print(f"- Test Accuracy: {results['accuracy']:.4f}")
        print(f"- Test F1-Score: {results['f1_score']:.4f}")
        print(f"- Model saved to: {model_path}")
        print("- Visualizations saved to plots/ directory")
        
    except Exception as e:
        print(f"\\nError during training: {e}")
        raise e

if __name__ == "__main__":
    main()
