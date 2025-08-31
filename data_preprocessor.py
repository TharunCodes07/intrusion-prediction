"""
Data preprocessing utilities for MSAFN Network Intrusion Detection System
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
from config import Config

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
    def load_data(self):
        """Load and combine both datasets"""
        print("Loading datasets...")
        
        # Load infiltration data
        df1 = pd.read_csv(Config.INFILTRATION_DATA)
        print(f"Infiltration data shape: {df1.shape}")
        
        # Load web attacks data
        df2 = pd.read_csv(Config.WEBATTACKS_DATA)
        print(f"Web attacks data shape: {df2.shape}")
        
        # Combine datasets
        df = pd.concat([df1, df2], ignore_index=True)
        print(f"Combined data shape: {df.shape}")
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        return df
    
    def clean_data(self, df):
        """Clean and preprocess the data"""
        print("Cleaning data...")
        
        # Handle missing values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Extract features and labels
        label_col = 'Label'
        X = df.drop(columns=[label_col])
        y = df[label_col].str.strip()
        
        self.feature_names = X.columns.tolist()
        
        print(f"Features shape: {X.shape}")
        print(f"Label distribution:")
        print(y.value_counts())
        
        return X, y
    
    def encode_labels(self, y):
        """Encode labels to numerical format"""
        # Map labels using config mapping
        y_mapped = y.map(Config.ATTACK_MAPPING)
        
        # Handle any unmapped labels
        if y_mapped.isna().any():
            print("Warning: Some labels not found in mapping. Using label encoder fallback.")
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = y_mapped.values
            
        return y_encoded
    
    def create_sequences(self, X, y, sequence_length=Config.SEQUENCE_LENGTH):
        """Create sequences for temporal modeling"""
        sequences = []
        labels = []
        
        # Group by similar flow characteristics for sequence creation
        # For simplicity, we'll create overlapping windows
        for i in range(len(X) - sequence_length + 1):
            sequences.append(X[i:i + sequence_length])
            labels.append(y[i + sequence_length - 1])  # Use the last label in sequence
            
        return np.array(sequences), np.array(labels)
    
    def gpu_balance_data(self, X, y, max_samples_per_class=50000):
        """GPU-accelerated data balancing with memory optimization"""
        print("Applying GPU-accelerated data balancing...")
        
        # Convert to tensors for GPU operations
        X_tensor = tf.constant(X, dtype=tf.float32)
        y_tensor = tf.constant(y, dtype=tf.int32)
        
        # Get unique classes and their counts
        unique_labels, _, counts = tf.unique_with_counts(y_tensor)
        unique_labels_np = unique_labels.numpy()
        counts_np = counts.numpy()
        
        print(f"Original class distribution: {dict(zip(unique_labels_np, counts_np))}")
        
        # Set target count (use min of max_count and max_samples_per_class)
        max_count = tf.reduce_max(counts)
        target_count = min(max_count.numpy(), max_samples_per_class)
        
        print(f"Target samples per class: {target_count}")
        
        # Balance each class to match the target count
        balanced_X_list = []
        balanced_y_list = []
        
        for i, label in enumerate(unique_labels_np):
            # Get indices for this class
            class_mask = tf.equal(y_tensor, label)
            class_indices = tf.where(class_mask)[:, 0]
            class_X = tf.gather(X_tensor, class_indices)
            class_y = tf.gather(y_tensor, class_indices)
            
            current_count = counts_np[i]
            
            if current_count < target_count:
                # Calculate how many samples we need to generate
                samples_needed = target_count - current_count
                
                # Random oversampling with noise injection
                indices_to_repeat = tf.random.uniform([samples_needed], 0, current_count, dtype=tf.int32)
                repeated_X = tf.gather(class_X, indices_to_repeat)
                repeated_y = tf.gather(class_y, indices_to_repeat)
                
                # Add small amount of Gaussian noise
                noise = tf.random.normal(tf.shape(repeated_X), mean=0.0, stddev=0.005)
                repeated_X = repeated_X + noise
                
                # Combine original and synthetic samples
                combined_X = tf.concat([class_X, repeated_X], axis=0)
                combined_y = tf.concat([class_y, repeated_y], axis=0)
            elif current_count > target_count:
                # Downsample if class is too large
                indices_to_keep = tf.random.uniform([target_count], 0, current_count, dtype=tf.int32)
                combined_X = tf.gather(class_X, indices_to_keep)
                combined_y = tf.gather(class_y, indices_to_keep)
            else:
                combined_X = class_X
                combined_y = class_y
            
            balanced_X_list.append(combined_X)
            balanced_y_list.append(combined_y)
        
        # Concatenate all balanced classes
        balanced_X = tf.concat(balanced_X_list, axis=0)
        balanced_y = tf.concat(balanced_y_list, axis=0)
        
        # Shuffle the balanced dataset
        indices = tf.range(tf.shape(balanced_X)[0])
        shuffled_indices = tf.random.shuffle(indices)
        balanced_X = tf.gather(balanced_X, shuffled_indices)
        balanced_y = tf.gather(balanced_y, shuffled_indices)
        
        # Convert back to numpy
        balanced_X_np = balanced_X.numpy()
        balanced_y_np = balanced_y.numpy()
        
        # Print final distribution
        unique_final, counts_final = np.unique(balanced_y_np, return_counts=True)
        print(f"Balanced class distribution: {dict(zip(unique_final, counts_final))}")
        print(f"After balancing - X shape: {balanced_X_np.shape}")
        
        return balanced_X_np, balanced_y_np
    
    def prepare_data(self, apply_balancing=True):
        """Complete data preparation pipeline"""
        # Load and clean data
        df = self.load_data()
        X, y = self.clean_data(df)
        
        # Encode labels
        y_encoded = self.encode_labels(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply balancing if requested
        if apply_balancing:
            X_scaled, y_encoded = self.gpu_balance_data(X_scaled, y_encoded)
        
        return X_scaled, y_encoded
    
    def prepare_progressive_data(self):
        """Prepare data for progressive training"""
        df = self.load_data()
        X, y = self.clean_data(df)
        y_encoded = self.encode_labels(y)
        X_scaled = self.scaler.fit_transform(X)
        
        # Separate normal and attack data
        normal_mask = y_encoded == 0  # BENIGN
        attack_mask = ~normal_mask
        
        X_normal = X_scaled[normal_mask]
        y_normal = y_encoded[normal_mask]
        
        X_attack = X_scaled[attack_mask]
        y_attack = y_encoded[attack_mask]
        
        print(f"Normal samples: {len(X_normal)}")
        print(f"Attack samples: {len(X_attack)}")
        
        return (X_normal, y_normal), (X_attack, y_attack), (X_scaled, y_encoded)
    
    def split_data(self, X, y, test_size=Config.TEST_SPLIT, val_size=Config.VALIDATION_SPLIT):
        """Split data into train, validation, and test sets"""
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_tf_dataset(self, X, y, batch_size=Config.BATCH_SIZE, shuffle_data=True):
        """Create TensorFlow dataset"""
        dataset = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), y.astype(np.int32)))
        
        if shuffle_data:
            dataset = dataset.shuffle(buffer_size=10000)
            
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
