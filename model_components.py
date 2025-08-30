import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import logging

class TemporalStream(layers.Layer):
    """
    Temporal stream: 1D CNN followed by BiLSTM for temporal pattern extraction
    """
    
    def __init__(self, name="temporal_stream", **kwargs):
        super(TemporalStream, self).__init__(name=name, **kwargs)
        
        # 1D Convolutional layers
        self.conv1 = layers.Conv1D(64, 3, activation='relu', padding='same')
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.2)
        
        self.conv2 = layers.Conv1D(32, 3, activation='relu', padding='same')
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(0.2)
        
        # BiLSTM layers
        self.bilstm1 = layers.Bidirectional(layers.LSTM(64, return_sequences=True))
        self.dropout3 = layers.Dropout(0.3)
        
        self.bilstm2 = layers.Bidirectional(layers.LSTM(32, return_sequences=False))
        self.dropout4 = layers.Dropout(0.3)
        
        # Dense layer for final temporal representation
        self.dense = layers.Dense(64, activation='relu')
        self.bn3 = layers.BatchNormalization()
    
    def call(self, inputs, training=None):
        # Reshape for 1D convolution if needed
        if len(inputs.shape) == 2:
            x = tf.expand_dims(inputs, axis=-1)
        else:
            x = inputs
        
        # 1D CNN layers
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.dropout2(x, training=training)
        
        # BiLSTM layers
        x = self.bilstm1(x, training=training)
        x = self.dropout3(x, training=training)
        
        x = self.bilstm2(x, training=training)
        x = self.dropout4(x, training=training)
        
        # Final dense layer
        x = self.dense(x)
        x = self.bn3(x, training=training)
        
        return x

class StatisticalStream(layers.Layer):
    """
    Statistical stream: Dense layers with batch normalization for statistical features
    """
    
    def __init__(self, name="statistical_stream", **kwargs):
        super(StatisticalStream, self).__init__(name=name, **kwargs)
        
        self.dense1 = layers.Dense(128, activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.3)
        
        self.dense2 = layers.Dense(64, activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(0.3)
        
        self.dense3 = layers.Dense(32, activation='relu')
        self.bn3 = layers.BatchNormalization()
        self.dropout3 = layers.Dropout(0.2)
        
        self.output_dense = layers.Dense(64, activation='relu')
        self.output_bn = layers.BatchNormalization()
    
    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)
        
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = self.dropout2(x, training=training)
        
        x = self.dense3(x)
        x = self.bn3(x, training=training)
        x = self.dropout3(x, training=training)
        
        x = self.output_dense(x)
        x = self.output_bn(x, training=training)
        
        return x

class BehavioralStream(layers.Layer):
    """
    Behavioral stream: GRU with self-attention for behavioral pattern modeling
    """
    
    def __init__(self, name="behavioral_stream", **kwargs):
        super(BehavioralStream, self).__init__(name=name, **kwargs)
        
        # GRU layers
        self.gru1 = layers.GRU(64, return_sequences=True)
        self.dropout1 = layers.Dropout(0.3)
        
        self.gru2 = layers.GRU(32, return_sequences=True)
        self.dropout2 = layers.Dropout(0.3)
        
        # Self-attention mechanism
        self.attention = layers.MultiHeadAttention(
            num_heads=4, key_dim=32, name="self_attention"
        )
        self.layer_norm = layers.LayerNormalization()
        
        # Global pooling and dense
        self.global_pool = layers.GlobalAveragePooling1D()
        self.dense = layers.Dense(64, activation='relu')
        self.bn = layers.BatchNormalization()
    
    def call(self, inputs, training=None):
        # Reshape for sequence processing if needed
        if len(inputs.shape) == 2:
            x = tf.expand_dims(inputs, axis=1)
        else:
            x = inputs
        
        # GRU layers
        x = self.gru1(x, training=training)
        x = self.dropout1(x, training=training)
        
        x = self.gru2(x, training=training)
        x = self.dropout2(x, training=training)
        
        # Self-attention
        attention_output = self.attention(x, x, training=training)
        x = self.layer_norm(x + attention_output)
        
        # Global pooling and final dense
        x = self.global_pool(x)
        x = self.dense(x)
        x = self.bn(x, training=training)
        
        return x

class MultiHeadAttentionFusion(layers.Layer):
    """
    Multi-head attention fusion layer for combining multiple streams
    """
    
    def __init__(self, num_heads=8, key_dim=64, name="attention_fusion", **kwargs):
        super(MultiHeadAttentionFusion, self).__init__(name=name, **kwargs)
        
        self.num_heads = num_heads
        self.key_dim = key_dim
        
        # Multi-head attention for fusion
        self.cross_attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim
        )
        
        # Layer normalization and dense layers
        self.layer_norm1 = layers.LayerNormalization()
        self.layer_norm2 = layers.LayerNormalization()
        
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout1 = layers.Dropout(0.3)
        self.dense2 = layers.Dense(64, activation='relu')
        self.dropout2 = layers.Dropout(0.2)
        
        # Feature importance weights
        self.importance_weights = layers.Dense(3, activation='softmax', name='stream_weights')
    
    def call(self, streams, training=None):
        # streams is a list of [temporal, statistical, behavioral] outputs
        temporal, statistical, behavioral = streams
        
        # Stack streams for attention computation
        stacked_streams = tf.stack([temporal, statistical, behavioral], axis=1)
        
        # Apply cross-attention
        attention_output = self.cross_attention(
            stacked_streams, stacked_streams, training=training
        )
        attention_output = self.layer_norm1(stacked_streams + attention_output)
        
        # Compute importance weights
        weights = self.importance_weights(tf.reduce_mean(attention_output, axis=1))
        weights = tf.expand_dims(weights, axis=-1)
        
        # Weighted fusion
        weighted_streams = attention_output * weights
        fused_output = tf.reduce_sum(weighted_streams, axis=1)
        
        # Final processing
        x = self.dense1(fused_output)
        x = self.dropout1(x, training=training)
        x = self.layer_norm2(x)
        
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        
        return x, weights

class MSAFNModel(Model):
    """
    Multi-Stream Attention-Based Fusion Network for Network Intrusion Detection
    """
    
    def __init__(self, feature_groups, num_classes, name="msafn_model", **kwargs):
        super(MSAFNModel, self).__init__(name=name, **kwargs)
        
        self.feature_groups = feature_groups
        self.num_classes = num_classes
        
        # Feature extraction layers for each group
        self.temporal_extractor = layers.Dense(64, activation='relu', name='temporal_extractor')
        self.statistical_extractor = layers.Dense(64, activation='relu', name='statistical_extractor')
        self.behavioral_extractor = layers.Dense(64, activation='relu', name='behavioral_extractor')
        
        # Stream processing layers
        self.temporal_stream = TemporalStream()
        self.statistical_stream = StatisticalStream()
        self.behavioral_stream = BehavioralStream()
        
        # Fusion layer
        self.fusion_layer = MultiHeadAttentionFusion()
        
        # Classification head
        self.classifier = layers.Dense(num_classes, activation='softmax', name='classifier')
        
        # Store attention weights for visualization
        self.attention_weights = None
    
    def call(self, inputs, training=None):
        # Extract features for each stream
        temporal_features = tf.gather(inputs, self.feature_groups['temporal'], axis=1)
        statistical_features = tf.gather(inputs, self.feature_groups['statistical'], axis=1)
        behavioral_features = tf.gather(inputs, self.feature_groups['behavioral'], axis=1)
        
        # Initial feature extraction
        temporal_features = self.temporal_extractor(temporal_features)
        statistical_features = self.statistical_extractor(statistical_features)
        behavioral_features = self.behavioral_extractor(behavioral_features)
        
        # Process through specialized streams
        temporal_output = self.temporal_stream(temporal_features, training=training)
        statistical_output = self.statistical_stream(statistical_features, training=training)
        behavioral_output = self.behavioral_stream(behavioral_features, training=training)
        
        # Fusion with attention
        fused_output, attention_weights = self.fusion_layer(
            [temporal_output, statistical_output, behavioral_output], 
            training=training
        )
        
        # Store attention weights for visualization
        self.attention_weights = attention_weights
        
        # Final classification
        output = self.classifier(fused_output)
        
        return output
    
    def get_attention_weights(self):
        """Get the latest attention weights for visualization"""
        return self.attention_weights

def create_callbacks(model_save_path, log_dir, patience=15):
    """
    Create training callbacks for model training
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience//2,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        )
    ]
    
    return callbacks

def create_custom_loss(class_weights=None):
    """
    Create custom loss function for imbalanced classes
    """
    def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
        """
        Focal loss for addressing class imbalance
        """
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Convert labels to one-hot if needed
        if len(y_true.shape) == 1 or y_true.shape[-1] == 1:
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), tf.shape(y_pred)[-1])
        
        # Calculate focal loss
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        
        focal_loss = -alpha_t * tf.pow(1 - pt, gamma) * tf.log(pt)
        
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=1))
    
    if class_weights is not None:
        def weighted_categorical_crossentropy(y_true, y_pred):
            """
            Weighted categorical crossentropy
            """
            # Convert to one-hot if needed
            if len(y_true.shape) == 1 or y_true.shape[-1] == 1:
                y_true = tf.one_hot(tf.cast(y_true, tf.int32), tf.shape(y_pred)[-1])
            
            # Apply class weights
            weights = tf.gather(class_weights, tf.argmax(y_true, axis=1))
            weights = tf.cast(weights, tf.float32)
            
            # Calculate weighted loss
            loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            weighted_loss = loss * weights
            
            return tf.reduce_mean(weighted_loss)
        
        return weighted_categorical_crossentropy
    else:
        return focal_loss

def create_metrics():
    """
    Create comprehensive metrics for evaluation
    """
    metrics = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.AUC(name='pr_auc', curve='PR'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')
    ]
    
    return metrics
