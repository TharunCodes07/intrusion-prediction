# Network Intrusion Detection with Multi-Stream Attention-Based Fusion Network (MSAFN)

This project implements a sophisticated hybrid deep learning approach for network intrusion detection using Multi-Stream Attention-Based Fusion Networks.

## Dataset

- **Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv**: 288,566 BENIGN + 36 Infiltration attacks
- **Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv**: 168,186 BENIGN + 2,180 Web attacks
- **Total Features**: 78 network flow features
- **Total Samples**: ~459K

## Architecture Overview

The MSAFN consists of three specialized streams:

1. **Temporal Stream**: 1D CNN → BiLSTM for temporal patterns
2. **Statistical Stream**: Dense layers with batch normalization for statistical features
3. **Behavioral Stream**: GRU with self-attention for behavioral patterns
4. **Multi-Head Attention Fusion**: Learns optimal feature combination weights

## Implementation

- `msafn_standard.py`: Standard training approach
- `msafn_progressive.py`: Progressive training (normal first, then attacks)
- `data_preprocessing.py`: Data loading and preprocessing utilities
- `model_components.py`: Reusable model components
- `attention_visualization.py`: Attention weight visualization
- `utils.py`: Utility functions for logging and evaluation

## Training Approaches

1. **Standard Training**: Train on mixed normal and attack data
2. **Progressive Training**: Start with normal traffic, progressively add attack types

## Features

- Comprehensive logging with TensorBoard
- Model checkpointing and early stopping
- Attention visualization
- Class imbalance handling
- Robust evaluation metrics
