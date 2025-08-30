import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
import logging
import os

class NetworkDataPreprocessor:
    """
    Comprehensive data preprocessing for network intrusion detection
    """
    
    def __init__(self, log_dir='logs'):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging with UTF-8 encoding
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{log_dir}/preprocessing.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, file_paths):
        """
        Load and combine multiple CSV files
        """
        self.logger.info(f"Loading data from {len(file_paths)} files")
        
        dataframes = []
        for file_path in file_paths:
            try:
                df = pd.read_csv(file_path)
                self.logger.info(f"Loaded {file_path}: {df.shape[0]} rows, {df.shape[1]} columns")
                dataframes.append(df)
            except Exception as e:
                self.logger.error(f"Error loading {file_path}: {str(e)}")
                raise
        
        # Combine all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        self.logger.info(f"Combined dataset: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")
        
        return combined_df
    
    def clean_data(self, df):
        """
        Clean the dataset by handling missing values and infinite values
        """
        self.logger.info("Starting data cleaning...")
        
        # Remove leading/trailing spaces from column names
        df.columns = df.columns.str.strip()
        
        # Clean label column - handle Unicode characters
        if 'Label' in df.columns:
            # Replace problematic Unicode characters in labels
            df['Label'] = df['Label'].str.replace('�', '-', regex=False)
            df['Label'] = df['Label'].str.replace('–', '-', regex=False)  # en-dash
            df['Label'] = df['Label'].str.replace('—', '-', regex=False)  # em-dash
            
            # Log the cleaned labels
            unique_labels = df['Label'].unique()
            self.logger.info(f"Cleaned unique labels: {unique_labels}")
        
        # Handle infinite values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Log missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            self.logger.warning(f"Found {missing_counts.sum()} missing values")
            # Fill missing values with median for numerical columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # Remove duplicate rows
        initial_rows = len(df)
        df.drop_duplicates(inplace=True)
        removed_duplicates = initial_rows - len(df)
        if removed_duplicates > 0:
            self.logger.info(f"Removed {removed_duplicates} duplicate rows")
        
        self.logger.info("Data cleaning completed")
        return df
    
    def analyze_labels(self, df, label_column='Label'):
        """
        Analyze label distribution
        """
        self.logger.info("Analyzing label distribution...")
        
        label_counts = df[label_column].value_counts()
        self.logger.info(f"Label distribution:\n{label_counts}")
        
        # Calculate class imbalance ratio
        majority_class_count = label_counts.max()
        minority_class_count = label_counts.min()
        imbalance_ratio = majority_class_count / minority_class_count
        self.logger.info(f"Class imbalance ratio: {imbalance_ratio:.2f}")
        
        return label_counts
    
    def separate_features_labels(self, df, label_column='Label'):
        """
        Separate features and labels
        """
        # Ensure label column exists
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in dataset")
        
        X = df.drop(columns=[label_column])
        y = df[label_column]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        self.logger.info(f"Features shape: {X.shape}")
        self.logger.info(f"Labels shape: {y.shape}")
        
        return X, y
    
    def encode_labels(self, y):
        """
        Encode categorical labels to numerical
        """
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Log label mapping
        label_mapping = dict(zip(self.label_encoder.classes_, 
                               self.label_encoder.transform(self.label_encoder.classes_)))
        self.logger.info(f"Label encoding mapping: {label_mapping}")
        
        return y_encoded
    
    def scale_features(self, X_train, X_test=None):
        """
        Scale features using StandardScaler
        """
        self.logger.info("Scaling features...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def handle_class_imbalance(self, X, y, method='smote_tomek', random_state=42):
        """
        Handle class imbalance using various techniques
        """
        self.logger.info(f"Handling class imbalance using {method}...")
        
        original_distribution = np.bincount(y)
        self.logger.info(f"Original distribution: {original_distribution}")
        
        if method == 'smote':
            sampler = SMOTE(random_state=random_state)
        elif method == 'tomek':
            sampler = TomekLinks()
        elif method == 'smote_tomek':
            sampler = SMOTETomek(random_state=random_state)
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        new_distribution = np.bincount(y_resampled)
        self.logger.info(f"New distribution: {new_distribution}")
        
        return X_resampled, y_resampled
    
    def create_progressive_datasets(self, X, y, test_size=0.2, val_size=0.1, random_state=42):
        """
        Create datasets for progressive training
        Returns: normal_only, incremental_datasets, full_test_set
        """
        self.logger.info("Creating progressive training datasets...")
        
        # Convert to DataFrame for easier manipulation
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.feature_names)
        else:
            X_df = X.copy()
        
        y_df = pd.Series(y)
        
        # Combine for easier splitting
        combined_df = X_df.copy()
        combined_df['Label'] = y_df
        
        # Split into train and test first
        train_df, test_df = train_test_split(
            combined_df, test_size=test_size, 
            stratify=y_df, random_state=random_state
        )
        
        # Extract test set features and labels
        X_test = test_df.drop('Label', axis=1).values
        y_test = test_df['Label'].values
        
        # Work with training data for progressive training
        X_train_df = train_df.drop('Label', axis=1)
        y_train_df = train_df['Label']
        
        # Create normal-only dataset
        normal_mask = y_train_df == 0  # Assuming 0 is BENIGN
        X_normal = X_train_df[normal_mask].values
        y_normal = y_train_df[normal_mask].values
        
        # Split normal data into train and validation
        X_normal_train, X_normal_val, y_normal_train, y_normal_val = train_test_split(
            X_normal, y_normal, test_size=val_size, random_state=random_state
        )
        
        # Create incremental datasets (normal + each attack type)
        attack_types = np.unique(y_train_df[y_train_df != 0])
        incremental_datasets = []
        
        for i, attack_type in enumerate(attack_types):
            # Get data up to this attack type
            include_mask = (y_train_df == 0) | (y_train_df.isin(attack_types[:i+1]))
            X_incremental = X_train_df[include_mask].values
            y_incremental = y_train_df[include_mask].values
            
            # Split into train and validation
            X_inc_train, X_inc_val, y_inc_train, y_inc_val = train_test_split(
                X_incremental, y_incremental, test_size=val_size, 
                stratify=y_incremental, random_state=random_state
            )
            
            incremental_datasets.append({
                'X_train': X_inc_train,
                'y_train': y_inc_train,
                'X_val': X_inc_val,
                'y_val': y_inc_val,
                'attack_types': attack_types[:i+1].tolist()
            })
        
        self.logger.info(f"Created progressive datasets:")
        self.logger.info(f"- Normal only: {X_normal_train.shape[0]} train, {X_normal_val.shape[0]} val")
        for i, dataset in enumerate(incremental_datasets):
            self.logger.info(f"- Stage {i+1}: {dataset['X_train'].shape[0]} train, {dataset['X_val'].shape[0]} val")
        
        return {
            'normal_train': (X_normal_train, y_normal_train),
            'normal_val': (X_normal_val, y_normal_val),
            'incremental_datasets': incremental_datasets,
            'test': (X_test, y_test)
        }
    
    def prepare_standard_datasets(self, X, y, test_size=0.2, val_size=0.1, random_state=42):
        """
        Prepare datasets for standard training approach
        """
        self.logger.info("Preparing datasets for standard training...")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            stratify=y_temp, random_state=random_state
        )
        
        self.logger.info(f"Dataset splits:")
        self.logger.info(f"- Train: {X_train.shape[0]} samples")
        self.logger.info(f"- Validation: {X_val.shape[0]} samples")
        self.logger.info(f"- Test: {X_test.shape[0]} samples")
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
    
    def get_feature_groups(self):
        """
        Group features by type for multi-stream architecture
        """
        if self.feature_names is None:
            raise ValueError("Feature names not available. Call separate_features_labels first.")
        
        # Define feature groups based on network flow characteristics
        temporal_features = [
            'Flow Duration', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
            'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
            'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
            'Active Mean', 'Active Std', 'Active Max', 'Active Min',
            'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
        ]
        
        statistical_features = [
            'Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets',
            'Total Length of Bwd Packets', 'Fwd Packet Length Max', 'Fwd Packet Length Min',
            'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max',
            'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
            'Flow Bytes/s', 'Flow Packets/s', 'Min Packet Length', 'Max Packet Length',
            'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
            'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size'
        ]
        
        behavioral_features = [
            'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
            'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
            'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count',
            'Down/Up Ratio', 'Fwd Header Length', 'Bwd Header Length',
            'Fwd Packets/s', 'Bwd Packets/s', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
            'Subflow Bwd Packets', 'Subflow Bwd Bytes'
        ]
        
        # Find indices for each group
        temporal_indices = [i for i, name in enumerate(self.feature_names) 
                          if any(tf.strip() in name.strip() for tf in temporal_features)]
        statistical_indices = [i for i, name in enumerate(self.feature_names) 
                             if any(sf.strip() in name.strip() for sf in statistical_features)]
        behavioral_indices = [i for i, name in enumerate(self.feature_names) 
                            if any(bf.strip() in name.strip() for bf in behavioral_features)]
        
        # Remaining features go to statistical group
        all_assigned = set(temporal_indices + statistical_indices + behavioral_indices)
        remaining_indices = [i for i in range(len(self.feature_names)) if i not in all_assigned]
        statistical_indices.extend(remaining_indices)
        
        feature_groups = {
            'temporal': temporal_indices,
            'statistical': statistical_indices,
            'behavioral': behavioral_indices
        }
        
        self.logger.info(f"Feature groups created:")
        self.logger.info(f"- Temporal: {len(temporal_indices)} features")
        self.logger.info(f"- Statistical: {len(statistical_indices)} features")
        self.logger.info(f"- Behavioral: {len(behavioral_indices)} features")
        
        return feature_groups
