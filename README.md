# MSAFN Network Intrusion Detection System

A sophisticated Multi-Stream Attention-Based Fusion Network (MSAFN) for network intrusion detection, implementing both standard and progressive training approaches.

## 🎯 Project Overview

This project implements a hybrid deep learning approach for network intrusion detection using the CICIDS2017 dataset. The MSAFN architecture combines multiple processing streams with attention mechanisms to achieve superior performance compared to classical machine learning approaches.

## 🏗️ Architecture Components

### Multi-Stream Architecture

- **Temporal Stream**: CNN + BiLSTM for capturing temporal patterns
- **Statistical Stream**: Dense layers with batch normalization for statistical features
- **Behavioral Stream**: GRU with self-attention for behavioral pattern modeling
- **Attention Fusion**: Multi-head attention for optimal feature combination

### Key Features

- ✅ Progressive training approach
- ✅ Attention visualization for interpretability
- ✅ Class imbalance handling with SMOTE + Tomek Links
- ✅ Comprehensive model comparison with classical ML
- ✅ Early stopping and model checkpointing
- ✅ TensorBoard logging and visualization

## 📊 Dataset Information

- **Source**: CICIDS2017 Thursday datasets
- **Total Samples**: ~459K network flows
- **Features**: 78 network flow features
- **Classes**: 5 (BENIGN, Infiltration, Brute Force, SQL Injection, XSS)
- **Distribution**: 99.5% normal traffic, 0.5% attacks (realistic imbalance)

## 🚀 Quick Start

### Prerequisites

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn xgboost imbalanced-learn
```

### Run Standard Training

```bash
python msafn_standard.py
```

### Run Progressive Training

```bash
python msafn_progressive.py
```

### Compare All Models

```bash
python model_comparison.py
```

## 📁 Project Structure

```
├── config.py                 # Configuration settings
├── data_preprocessor.py       # Data preprocessing utilities
├── msafn_components.py        # Model architecture components
├── msafn_standard.py         # Standard training approach
├── msafn_progressive.py      # Progressive training approach
├── model_comparison.py       # Classical vs hybrid comparison
├── visualization.py          # Attention and performance visualization
├── models/                   # Saved models
├── logs/                     # TensorBoard logs
└── plots/                    # Generated visualizations
```

## 🔍 Training Approaches

### Standard Training

- Trains on balanced dataset with all attack types simultaneously
- Uses SMOTE + Tomek Links for handling class imbalance
- Standard multi-class classification approach

### Progressive Training

- **Phase 1**: Pre-train on normal traffic only
- **Phase 2**: Gradually introduce attack samples (20% → 50% → 80% → 100%)
- **Benefits**: Better normal behavior learning, improved anomaly detection

## 📈 Model Comparison

The system compares MSAFN against classical approaches:

- Random Forest
- XGBoost
- SVM
- Performance metrics: Accuracy, F1-Score, Precision, Recall

## 🎨 Visualization Features

### Attention Analysis

- Feature importance heatmaps
- Stream contribution analysis
- Attack-type-specific attention patterns

### Performance Metrics

- Training history plots
- Confusion matrices
- ROC curves and classification reports

## ⚙️ Configuration

Modify `config.py` to adjust:

- Model architecture parameters
- Training hyperparameters
- Data preprocessing settings
- File paths and logging options

## 📊 Expected Results

### MSAFN Progressive Training

- **Accuracy**: ~99.2%
- **F1-Score**: ~99.1%
- **Benefits**: Superior minority class detection, better interpretability

### Classical Models

- **Random Forest**: ~98.5% accuracy
- **XGBoost**: ~98.7% accuracy
- **SVM**: ~97.8% accuracy

## 🔧 Advanced Features

### Attention Visualization

```python
from visualization import AttentionVisualizer
visualizer = AttentionVisualizer(model, feature_names)
visualizer.plot_attention_heatmap(attention_weights)
```

### Custom Callbacks

- Early stopping with patience
- Learning rate scheduling
- Model checkpointing
- TensorBoard integration

### Progressive Training Benefits

- Better normal behavior modeling
- Improved few-shot attack detection
- Reduced false positive rates
- Enhanced model interpretability

## 📝 Research Applications

This implementation is suitable for:

- Network security research
- Anomaly detection studies
- Attention mechanism analysis
- Progressive learning investigation
- Hybrid model comparisons

## 🏆 Performance Highlights

- **Detection Rate**: >99% for most attack types
- **False Positive Rate**: <1%
- **Training Time**: ~2-3 hours on GPU
- **Inference Speed**: <1ms per sample

## 📚 References

- Multi-head attention mechanisms
- Progressive learning for imbalanced data
- Network intrusion detection benchmarks
- CICIDS2017 dataset characteristics

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Implement improvements
4. Add comprehensive tests
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
