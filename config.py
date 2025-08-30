"""
Configuration file for MSAFN Network Intrusion Detection System
"""

import os

class Config:
    # Data paths
    INFILTRATION_DATA = "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
    WEBATTACKS_DATA = "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
    
    # Model parameters
    SEQUENCE_LENGTH = 10  # For temporal modeling
    FEATURE_DIM = 78  # Number of features (excluding label)
    NUM_CLASSES = 5  # BENIGN, Infiltration, Brute Force, SQL Injection, XSS
    
    # Training parameters
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.2
    
    # Early stopping
    PATIENCE = 15
    MIN_DELTA = 0.001
    
    # Model architecture
    TEMPORAL_UNITS = 128
    STATISTICAL_UNITS = 256
    BEHAVIORAL_UNITS = 128
    ATTENTION_HEADS = 8
    DROPOUT_RATE = 0.3
    
    # Progressive training
    NORMAL_EPOCHS = 30
    PROGRESSIVE_EPOCHS = 50
    
    # Paths
    MODEL_SAVE_PATH = "models"
    LOGS_PATH = "logs"
    PLOTS_PATH = "plots"
    
    # Ensure directories exist
    for path in [MODEL_SAVE_PATH, LOGS_PATH, PLOTS_PATH]:
        os.makedirs(path, exist_ok=True)
    
    # Attack type mapping
    ATTACK_MAPPING = {
        'BENIGN': 0,
        'Infiltration': 1,
        'Web Attack � Brute Force': 2,
        'Web Attack � Sql Injection': 3,
        'Web Attack � XSS': 4
    }
