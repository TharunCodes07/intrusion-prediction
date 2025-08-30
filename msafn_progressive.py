"""
MSAFN Model Training - Progressive Training Approach
Multi-Stream Attention-Based Fusion Network for Network Intrusion Detection
Progressive training: Start with normal traffic, then gradually add attack samples
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
import os
import datetime

# Import custom modules
from config import Config
from data_preprocessor import DataPreprocessor
from msafn_components import build_msafn_model, AdversarialTraining
from visualization import AttentionVisualizer, ModelEvaluator

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class MSAFNProgressiveTrainer:
    """Progressive MSAFN trainer - starts with normal traffic, then adds attacks"""
    
    def __init__(self):
        self.model = None
        self.history_normal = None
        self.history_progressive = None
        self.preprocessor = DataPreprocessor()
        self.evaluator = ModelEvaluator()
        
    def prepare_progressive_data(self):
        """Prepare data for progressive training"""
        print("=" * 60)
        print("PREPARING DATA FOR PROGRESSIVE TRAINING")
        print("=" * 60)
        
        # Get separated normal and attack data
        (X_normal, y_normal), (X_attack, y_attack), (X_all, y_all) = self.preprocessor.prepare_progressive_data()
        
        # Split normal data for initial training
        from sklearn.model_selection import train_test_split
        
        X_normal_train, X_normal_val, y_normal_train, y_normal_val = train_test_split(
            X_normal, y_normal, test_size=0.3, random_state=42
        )
        
        # Split attack data
        X_attack_train, X_attack_val, y_attack_train, y_attack_val = train_test_split(
            X_attack, y_attack, test_size=0.3, random_state=42, stratify=y_attack
        )
        
        # Final test set (mixed normal + attack)
        X_test_final, _, y_test_final, _ = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
        )
        
        print(f"Normal training samples: {len(X_normal_train)}")
        print(f"Normal validation samples: {len(X_normal_val)}")
        print(f"Attack training samples: {len(X_attack_train)}")
        print(f"Attack validation samples: {len(X_attack_val)}")
        print(f"Final test samples: {len(X_test_final)}")
        
        return {
            'normal_train': (X_normal_train, y_normal_train),
            'normal_val': (X_normal_val, y_normal_val),
            'attack_train': (X_attack_train, y_attack_train),
            'attack_val': (X_attack_val, y_attack_val),
            'test_final': (X_test_final, y_test_final)
        }
    
    def build_model(self):
        """Build and compile the MSAFN model for progressive training"""
        print("\\nBuilding MSAFN Model for Progressive Training...")
        
        self.model = build_msafn_model()
        
        # Print model summary
        self.model.summary()
        
        # Initial compilation for normal traffic (binary classification: normal vs anomaly)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
            loss='sparse_categorical_crossentropy',  # Will change to categorical later
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.F1Score(name='f1_score')
            ]
        )
        
        return self.model
    
    def setup_callbacks(self, phase="normal"):
        """Setup training callbacks for different phases"""
        callbacks = []
        
        # Early stopping
        patience = Config.PATIENCE if phase == "progressive" else Config.PATIENCE // 2
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            min_delta=Config.MIN_DELTA,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Model checkpoint
        checkpoint_name = f'msafn_progressive_{phase}_best.keras'
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(Config.MODEL_SAVE_PATH, checkpoint_name),
            monitor='val_f1_score',
            mode='max',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(model_checkpoint)
        
        # Reduce learning rate on plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # TensorBoard logging
        log_dir = os.path.join(Config.LOGS_PATH, f"msafn_progressive_{phase}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        callbacks.append(tensorboard)
        
        return callbacks
    
    def train_phase1_normal(self, data_dict):
        """Phase 1: Train on normal traffic only"""
        print("\\n" + "=" * 60)
        print("PHASE 1: TRAINING ON NORMAL TRAFFIC ONLY")
        print("=" * 60)
        
        X_normal_train, y_normal_train = data_dict['normal_train']
        X_normal_val, y_normal_val = data_dict['normal_val']
        
        # For normal-only training, we create a simple binary task
        # All samples are labeled as 0 (normal)
        y_normal_train_binary = np.zeros_like(y_normal_train)
        y_normal_val_binary = np.zeros_like(y_normal_val)
        
        # Setup callbacks
        callbacks = self.setup_callbacks(phase="normal")
        
        # Train on normal data
        self.history_normal = self.model.fit(
            X_normal_train, y_normal_train_binary,
            batch_size=Config.BATCH_SIZE,
            epochs=Config.NORMAL_EPOCHS,
            validation_data=(X_normal_val, y_normal_val_binary),
            callbacks=callbacks,
            verbose=1
        )
        
        print("Phase 1 (Normal Traffic) Training Completed!")
        return self.history_normal
    
    def train_phase2_progressive(self, data_dict):
        """Phase 2: Progressive training with normal + attack data"""
        print("\\n" + "=" * 60)
        print("PHASE 2: PROGRESSIVE TRAINING WITH ATTACKS")
        print("=" * 60)
        
        X_normal_train, y_normal_train = data_dict['normal_train']
        X_normal_val, y_normal_val = data_dict['normal_val']
        X_attack_train, y_attack_train = data_dict['attack_train']
        X_attack_val, y_attack_val = data_dict['attack_val']
        
        # Combine normal and attack data
        X_combined_train = np.vstack([X_normal_train, X_attack_train])
        y_combined_train = np.hstack([y_normal_train, y_attack_train])
        
        X_combined_val = np.vstack([X_normal_val, X_attack_val])
        y_combined_val = np.hstack([y_normal_val, y_attack_val])
        
        # Shuffle combined data
        X_combined_train, y_combined_train = shuffle(X_combined_train, y_combined_train, random_state=42)
        X_combined_val, y_combined_val = shuffle(X_combined_val, y_combined_val, random_state=42)
        
        # Convert to categorical for multi-class classification
        y_combined_train_cat = tf.keras.utils.to_categorical(y_combined_train, num_classes=Config.NUM_CLASSES)
        y_combined_val_cat = tf.keras.utils.to_categorical(y_combined_val, num_classes=Config.NUM_CLASSES)
        
        # Recompile model for multi-class classification
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE * 0.1),  # Lower LR
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.F1Score(name='f1_score')
            ]
        )
        
        # Gradual introduction of attack samples
        attack_ratios = [0.2, 0.5, 0.8, 1.0]  # Gradually increase attack sample ratio
        
        for i, ratio in enumerate(attack_ratios):
            print(f"\\n--- Sub-phase 2.{i+1}: Attack ratio = {ratio:.1f} ---")
            
            # Select subset of attack samples
            n_attack_samples = int(len(X_attack_train) * ratio)
            attack_indices = np.random.choice(len(X_attack_train), n_attack_samples, replace=False)
            
            X_attack_subset = X_attack_train[attack_indices]
            y_attack_subset = y_attack_train[attack_indices]
            
            # Combine with normal data
            X_progressive = np.vstack([X_normal_train, X_attack_subset])
            y_progressive = np.hstack([y_normal_train, y_attack_subset])
            
            # Shuffle
            X_progressive, y_progressive = shuffle(X_progressive, y_progressive, random_state=42)
            
            # Convert to categorical
            y_progressive_cat = tf.keras.utils.to_categorical(y_progressive, num_classes=Config.NUM_CLASSES)
            
            # Train for fewer epochs in each sub-phase
            epochs_subphase = Config.PROGRESSIVE_EPOCHS // len(attack_ratios)
            
            history_subphase = self.model.fit(
                X_progressive, y_progressive_cat,
                batch_size=Config.BATCH_SIZE,
                epochs=epochs_subphase,
                validation_data=(X_combined_val, y_combined_val_cat),
                verbose=1
            )
        
        # Final training phase with all data
        print("\\n--- Final Progressive Training Phase ---")
        callbacks = self.setup_callbacks(phase="progressive")
        
        self.history_progressive = self.model.fit(
            X_combined_train, y_combined_train_cat,
            batch_size=Config.BATCH_SIZE,
            epochs=Config.PROGRESSIVE_EPOCHS,
            validation_data=(X_combined_val, y_combined_val_cat),
            callbacks=callbacks,
            verbose=1
        )
        
        print("Phase 2 (Progressive Training) Completed!")
        return self.history_progressive
    
    def evaluate_model(self, data_dict):
        """Comprehensive model evaluation"""
        print("\\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        
        X_test, y_test = data_dict['test_final']
        
        # Convert to categorical for evaluation
        y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=Config.NUM_CLASSES)
        
        # Predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Evaluation metrics
        test_loss, test_accuracy, test_precision, test_recall, test_f1 = self.model.evaluate(
            X_test, y_test_cat, verbose=0
        )
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test F1-Score: {test_f1:.4f}")
        
        # Classification report
        report = self.evaluator.generate_classification_report(y_test, y_pred)
        
        # Confusion matrix
        self.evaluator.plot_confusion_matrix(
            y_test, y_pred,
            save_path=os.path.join(Config.PLOTS_PATH, 'confusion_matrix_progressive.png')
        )
        
        # Training history plots
        if self.history_normal:
            plt.figure(figsize=(15, 10))
            
            # Normal training history
            plt.subplot(2, 2, 1)
            plt.plot(self.history_normal.history['loss'], label='Normal Training Loss')
            plt.plot(self.history_normal.history['val_loss'], label='Normal Validation Loss')
            plt.title('Phase 1: Normal Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(2, 2, 2)
            plt.plot(self.history_normal.history['accuracy'], label='Normal Training Accuracy')
            plt.plot(self.history_normal.history['val_accuracy'], label='Normal Validation Accuracy')
            plt.title('Phase 1: Normal Training Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            # Progressive training history
            if self.history_progressive:
                plt.subplot(2, 2, 3)
                plt.plot(self.history_progressive.history['loss'], label='Progressive Training Loss')
                plt.plot(self.history_progressive.history['val_loss'], label='Progressive Validation Loss')
                plt.title('Phase 2: Progressive Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                
                plt.subplot(2, 2, 4)
                plt.plot(self.history_progressive.history['accuracy'], label='Progressive Training Accuracy')
                plt.plot(self.history_progressive.history['val_accuracy'], label='Progressive Validation Accuracy')
                plt.title('Phase 2: Progressive Training Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(Config.PLOTS_PATH, 'training_history_progressive.png'), dpi=300, bbox_inches='tight')
            plt.show()
        
        return {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': test_f1,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def visualize_attention(self, data_dict):
        """Visualize attention weights"""
        print("\\nGenerating Attention Visualizations...")
        
        X_test, y_test = data_dict['test_final']
        
        # Initialize visualizer
        visualizer = AttentionVisualizer(self.model, self.preprocessor.feature_names)
        
        # Plot attention heatmap
        try:
            behavioral_attention, fusion_attention = visualizer.extract_attention_weights(X_test[:100])
            
            visualizer.plot_attention_heatmap(
                fusion_attention,
                save_path=os.path.join(Config.PLOTS_PATH, 'attention_heatmap_progressive.png')
            )
            
            visualizer.plot_feature_attention_distribution(
                X_test[:100], y_test[:100],
                save_path=os.path.join(Config.PLOTS_PATH, 'feature_attention_progressive.png')
            )
            
            visualizer.plot_stream_contributions(
                X_test[:50],
                save_path=os.path.join(Config.PLOTS_PATH, 'stream_contributions_progressive.png')
            )
            
        except Exception as e:
            print(f"Warning: Could not generate attention visualizations: {e}")
    
    def save_model(self):
        """Save the trained model"""
        model_path = os.path.join(Config.MODEL_SAVE_PATH, 'msafn_progressive_final.keras')
        self.model.save(model_path)
        print(f"\\nModel saved to: {model_path}")
        
        # Save model architecture as JSON
        model_json = self.model.to_json()
        with open(os.path.join(Config.MODEL_SAVE_PATH, 'msafn_progressive_architecture.json'), 'w') as json_file:
            json_file.write(model_json)
        
        return model_path

def main():
    """Main progressive training pipeline"""
    print("MSAFN Network Intrusion Detection - Progressive Training")
    print("=" * 60)
    
    # Initialize trainer
    trainer = MSAFNProgressiveTrainer()
    
    try:
        # Prepare data
        data_dict = trainer.prepare_progressive_data()
        
        # Build model
        model = trainer.build_model()
        
        # Phase 1: Train on normal traffic
        history_normal = trainer.train_phase1_normal(data_dict)
        
        # Phase 2: Progressive training with attacks
        history_progressive = trainer.train_phase2_progressive(data_dict)
        
        # Evaluate model
        results = trainer.evaluate_model(data_dict)
        
        # Visualize attention
        trainer.visualize_attention(data_dict)
        
        # Save model
        model_path = trainer.save_model()
        
        print("\\n" + "=" * 60)
        print("PROGRESSIVE TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Final Results:")
        print(f"- Test Accuracy: {results['accuracy']:.4f}")
        print(f"- Test F1-Score: {results['f1_score']:.4f}")
        print(f"- Model saved to: {model_path}")
        print("- Visualizations saved to plots/ directory")
        
    except Exception as e:
        print(f"\\nError during progressive training: {e}")
        raise e

if __name__ == "__main__":
    main()
