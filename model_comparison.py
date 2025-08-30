"""
Model Comparison and Analysis
Compare MSAFN standard vs progressive training approaches and classical models
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import os
import joblib
import time

from config import Config
from data_preprocessor import DataPreprocessor
from visualization import ModelEvaluator

class ModelComparator:
    """Compare different models for network intrusion detection"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.evaluator = ModelEvaluator()
        self.results = {}
        
    def prepare_data_for_classical(self):
        """Prepare data for classical machine learning models"""
        print("Preparing data for classical models...")
        
        # Load and preprocess data
        X, y = self.preprocessor.prepare_data(apply_balancing=True)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.split_data(X, y)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_random_forest(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Train Random Forest classifier"""
        print("\\nTraining Random Forest...")
        
        start_time = time.time()
        
        # Random Forest with optimized parameters
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Train model
        rf_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = rf_model.predict(X_test)
        
        training_time = time.time() - start_time
        
        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Save model
        joblib.dump(rf_model, os.path.join(Config.MODEL_SAVE_PATH, 'random_forest_model.pkl'))
        
        self.results['Random Forest'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'training_time': training_time,
            'predictions': y_pred,
            'model': rf_model
        }
        
        print(f"Random Forest - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Time: {training_time:.2f}s")
        
        return rf_model
    
    def train_xgboost(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Train XGBoost classifier"""
        print("\\nTraining XGBoost...")
        
        start_time = time.time()
        
        # Calculate class weights for imbalanced data
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        weight_dict = dict(zip(classes, class_weights))
        
        # XGBoost with optimized parameters
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss',
            early_stopping_rounds=20
        )
        
        # Train model
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Predictions
        y_pred = xgb_model.predict(X_test)
        
        training_time = time.time() - start_time
        
        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Save model
        joblib.dump(xgb_model, os.path.join(Config.MODEL_SAVE_PATH, 'xgboost_model.pkl'))
        
        self.results['XGBoost'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'training_time': training_time,
            'predictions': y_pred,
            'model': xgb_model
        }
        
        print(f"XGBoost - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Time: {training_time:.2f}s")
        
        return xgb_model
    
    def train_svm(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Train SVM classifier"""
        print("\\nTraining SVM...")
        
        start_time = time.time()
        
        # Use subset for SVM due to computational complexity
        max_samples = 50000
        if len(X_train) > max_samples:
            indices = np.random.choice(len(X_train), max_samples, replace=False)
            X_train_svm = X_train[indices]
            y_train_svm = y_train[indices]
        else:
            X_train_svm = X_train
            y_train_svm = y_train
        
        # SVM with RBF kernel
        svm_model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight='balanced',
            random_state=42
        )
        
        # Train model
        svm_model.fit(X_train_svm, y_train_svm)
        
        # Predictions
        y_pred = svm_model.predict(X_test)
        
        training_time = time.time() - start_time
        
        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Save model
        joblib.dump(svm_model, os.path.join(Config.MODEL_SAVE_PATH, 'svm_model.pkl'))
        
        self.results['SVM'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'training_time': training_time,
            'predictions': y_pred,
            'model': svm_model
        }
        
        print(f"SVM - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Time: {training_time:.2f}s")
        
        return svm_model
    
    def load_deep_learning_results(self):
        """Load results from deep learning models"""
        print("\\nLoading Deep Learning Model Results...")
        
        # Try to load MSAFN models and their results
        try:
            # Load standard MSAFN model
            standard_model_path = os.path.join(Config.MODEL_SAVE_PATH, 'msafn_standard_final.keras')
            if os.path.exists(standard_model_path):
                standard_model = tf.keras.models.load_model(standard_model_path)
                print("Loaded MSAFN Standard model")
                
                # Evaluate on test data
                X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data_for_classical()
                y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=Config.NUM_CLASSES)
                
                # Predictions
                y_pred_proba = standard_model.predict(X_test)
                y_pred = np.argmax(y_pred_proba, axis=1)
                
                # Evaluation
                test_results = standard_model.evaluate(X_test, y_test_cat, verbose=0)
                accuracy = test_results[1]  # accuracy is second metric
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                self.results['MSAFN Standard'] = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'predictions': y_pred,
                    'model': standard_model
                }
                
                print(f"MSAFN Standard - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            # Load progressive MSAFN model
            progressive_model_path = os.path.join(Config.MODEL_SAVE_PATH, 'msafn_progressive_final.keras')
            if os.path.exists(progressive_model_path):
                progressive_model = tf.keras.models.load_model(progressive_model_path)
                print("Loaded MSAFN Progressive model")
                
                # Evaluate on test data
                X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data_for_classical()
                y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=Config.NUM_CLASSES)
                
                # Predictions
                y_pred_proba = progressive_model.predict(X_test)
                y_pred = np.argmax(y_pred_proba, axis=1)
                
                # Evaluation
                test_results = progressive_model.evaluate(X_test, y_test_cat, verbose=0)
                accuracy = test_results[1]  # accuracy is second metric
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                self.results['MSAFN Progressive'] = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'predictions': y_pred,
                    'model': progressive_model
                }
                
                print(f"MSAFN Progressive - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
                
        except Exception as e:
            print(f"Could not load deep learning models: {e}")
    
    def generate_comprehensive_comparison(self):
        """Generate comprehensive comparison of all models"""
        print("\\n" + "=" * 60)
        print("COMPREHENSIVE MODEL COMPARISON")
        print("=" * 60)
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'F1-Score': results['f1_score'],
                'Training Time (s)': results.get('training_time', 'N/A')
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('F1-Score', ascending=False)
        
        print("\\nModel Performance Ranking:")
        print("=" * 50)
        print(df_comparison.to_string(index=False))
        
        # Save comparison to CSV
        df_comparison.to_csv(os.path.join(Config.PLOTS_PATH, 'model_comparison.csv'), index=False)
        
        # Create visualization
        self.plot_model_comparison(df_comparison)
        
        return df_comparison
    
    def plot_model_comparison(self, df_comparison):
        """Plot model comparison charts"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy comparison
        axes[0, 0].bar(df_comparison['Model'], df_comparison['Accuracy'])
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # F1-Score comparison
        axes[0, 1].bar(df_comparison['Model'], df_comparison['F1-Score'])
        axes[0, 1].set_title('Model F1-Score Comparison')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Training time comparison (exclude N/A values)
        time_data = df_comparison[df_comparison['Training Time (s)'] != 'N/A']
        if not time_data.empty:
            axes[1, 0].bar(time_data['Model'], time_data['Training Time (s)'].astype(float))
            axes[1, 0].set_title('Training Time Comparison')
            axes[1, 0].set_ylabel('Time (seconds)')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].set_yscale('log')  # Log scale for time
        
        # Performance vs Complexity scatter plot
        complexity_scores = {'Random Forest': 2, 'XGBoost': 3, 'SVM': 4, 'MSAFN Standard': 8, 'MSAFN Progressive': 9}
        x_complexity = [complexity_scores.get(model, 5) for model in df_comparison['Model']]
        
        scatter = axes[1, 1].scatter(x_complexity, df_comparison['F1-Score'], 
                                   s=100, c=df_comparison['Accuracy'], cmap='viridis')
        axes[1, 1].set_title('Performance vs Model Complexity')
        axes[1, 1].set_xlabel('Model Complexity (Arbitrary Scale)')
        axes[1, 1].set_ylabel('F1-Score')
        
        # Add model names to scatter plot
        for i, model in enumerate(df_comparison['Model']):
            axes[1, 1].annotate(model[:10], (x_complexity[i], df_comparison['F1-Score'].iloc[i]),
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.colorbar(scatter, ax=axes[1, 1], label='Accuracy')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.PLOTS_PATH, 'comprehensive_model_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_feature_importance(self):
        """Analyze feature importance from different models"""
        print("\\nAnalyzing Feature Importance...")
        
        if 'Random Forest' in self.results:
            rf_model = self.results['Random Forest']['model']
            feature_importance = rf_model.feature_importances_
            
            # Get top 20 features
            top_indices = np.argsort(feature_importance)[-20:]
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(top_indices)), feature_importance[top_indices])
            plt.yticks(range(len(top_indices)), 
                      [self.preprocessor.feature_names[i] for i in top_indices])
            plt.xlabel('Feature Importance')
            plt.title('Top 20 Most Important Features (Random Forest)')
            plt.tight_layout()
            plt.savefig(os.path.join(Config.PLOTS_PATH, 'feature_importance_rf.png'), 
                       dpi=300, bbox_inches='tight')
            plt.show()

def main():
    """Main comparison pipeline"""
    print("Model Comparison Analysis")
    print("=" * 60)
    
    comparator = ModelComparator()
    
    try:
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = comparator.prepare_data_for_classical()
        
        # Train classical models
        comparator.train_random_forest(X_train, X_val, X_test, y_train, y_val, y_test)
        comparator.train_xgboost(X_train, X_val, X_test, y_train, y_val, y_test)
        comparator.train_svm(X_train, X_val, X_test, y_train, y_val, y_test)
        
        # Load deep learning results
        comparator.load_deep_learning_results()
        
        # Generate comprehensive comparison
        comparison_df = comparator.generate_comprehensive_comparison()
        
        # Analyze feature importance
        comparator.analyze_feature_importance()
        
        print("\\n" + "=" * 60)
        print("MODEL COMPARISON COMPLETED!")
        print("=" * 60)
        print("Results saved in:")
        print("- CSV: plots/model_comparison.csv")
        print("- Plots: plots/comprehensive_model_comparison.png")
        print("- Feature importance: plots/feature_importance_rf.png")
        
    except Exception as e:
        print(f"\\nError during model comparison: {e}")
        raise e

if __name__ == "__main__":
    main()
