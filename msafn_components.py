"""
MSAFN Model Components - Multi-Stream Attention-Based Fusion Network
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2
from config import Config

class MultiHeadAttention(layers.Layer):
    """Custom Multi-Head Attention layer"""
    def __init__(self, num_heads, d_model, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        
        self.dense = layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs):
        q, k, v = inputs, inputs, inputs
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))
        
        output = self.dense(concat_attention)
        return output, attention_weights

class TemporalStream(layers.Layer):
    """Temporal stream using CNN + BiLSTM"""
    def __init__(self, units=Config.TEMPORAL_UNITS, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        
        # 1D Convolutional layers
        self.conv1 = layers.Conv1D(64, 3, activation='relu', padding='same')
        self.conv2 = layers.Conv1D(128, 3, activation='relu', padding='same')
        self.pool = layers.MaxPooling1D(2)
        self.dropout1 = layers.Dropout(Config.DROPOUT_RATE)
        
        # BiLSTM layers
        self.bilstm = layers.Bidirectional(
            layers.LSTM(units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
        )
        self.dropout2 = layers.Dropout(Config.DROPOUT_RATE)
        
    def call(self, inputs, training=None):
        x = tf.expand_dims(inputs, axis=1)  # Add temporal dimension
        x = tf.tile(x, [1, Config.SEQUENCE_LENGTH, 1])  # Repeat for sequence
        
        # CNN processing
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout1(x, training=training)
        
        # BiLSTM processing
        x = self.bilstm(x)
        x = self.dropout2(x, training=training)
        
        return x

class StatisticalStream(layers.Layer):
    """Statistical stream using Dense layers with Batch Normalization"""
    def __init__(self, units=Config.STATISTICAL_UNITS, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        
        self.dense1 = layers.Dense(units, activation='relu', kernel_regularizer=l2(0.01))
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(Config.DROPOUT_RATE)
        
        self.dense2 = layers.Dense(units // 2, activation='relu', kernel_regularizer=l2(0.01))
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(Config.DROPOUT_RATE)
        
        self.dense3 = layers.Dense(units // 4, activation='relu')
        
    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)
        
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = self.dropout2(x, training=training)
        
        x = self.dense3(x)
        
        # Expand dims to match temporal stream output
        x = tf.expand_dims(x, axis=1)
        x = tf.tile(x, [1, Config.SEQUENCE_LENGTH // 2, 1])
        
        return x

class BehavioralStream(layers.Layer):
    """Behavioral stream using GRU with self-attention"""
    def __init__(self, units=Config.BEHAVIORAL_UNITS, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        
        # Feature transformation
        self.feature_transform = layers.Dense(units)
        
        # GRU layer
        self.gru = layers.GRU(units, return_sequences=True, dropout=0.2)
        
        # Self-attention
        self.attention = MultiHeadAttention(4, units)
        self.dropout = layers.Dropout(Config.DROPOUT_RATE)
        
    def call(self, inputs, training=None):
        # Transform input features
        x = self.feature_transform(inputs)
        
        # Create sequences for GRU
        x = tf.expand_dims(x, axis=1)
        x = tf.tile(x, [1, Config.SEQUENCE_LENGTH, 1])
        
        # GRU processing
        x = self.gru(x)
        
        # Self-attention
        attention_output, attention_weights = self.attention(x)
        x = self.dropout(attention_output, training=training)
        
        return x, attention_weights

class AttentionFusion(layers.Layer):
    """Multi-head attention fusion layer"""
    def __init__(self, d_model=256, num_heads=Config.ATTENTION_HEADS, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.multi_head_attention = MultiHeadAttention(num_heads, d_model)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.ffn = tf.keras.Sequential([
            layers.Dense(d_model * 2, activation='relu'),
            layers.Dropout(Config.DROPOUT_RATE),
            layers.Dense(d_model)
        ])
        
    def call(self, inputs, training=None):
        # Concatenate all streams
        fused_input = tf.concat(inputs, axis=-1)
        
        # Project to common dimension
        fused_input = layers.Dense(self.d_model)(fused_input)
        
        # Multi-head attention
        attn_output, attention_weights = self.multi_head_attention(fused_input)
        out1 = self.norm1(fused_input + attn_output)
        
        # Feed forward
        ffn_output = self.ffn(out1, training=training)
        out2 = self.norm2(out1 + ffn_output)
        
        return out2, attention_weights

def build_msafn_model(input_shape=(Config.FEATURE_DIM,), num_classes=Config.NUM_CLASSES):
    """Build the complete MSAFN model"""
    inputs = layers.Input(shape=input_shape, name='input_features')
    
    # Initialize streams
    temporal_stream = TemporalStream(name='temporal_stream')
    statistical_stream = StatisticalStream(name='statistical_stream')
    behavioral_stream = BehavioralStream(name='behavioral_stream')
    
    # Process through streams
    temporal_output = temporal_stream(inputs)
    statistical_output = statistical_stream(inputs)
    behavioral_output, behavioral_attention = behavioral_stream(inputs)
    
    # Fusion layer
    fusion_layer = AttentionFusion(name='attention_fusion')
    fused_output, fusion_attention = fusion_layer([temporal_output, statistical_output, behavioral_output])
    
    # Global pooling
    pooled_output = layers.GlobalAveragePooling1D()(fused_output)
    
    # Classification head
    x = layers.Dense(512, activation='relu', kernel_regularizer=l2(0.01))(pooled_output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(Config.DROPOUT_RATE)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(Config.DROPOUT_RATE)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='MSAFN')
    
    return model

class AdversarialTraining:
    """Adversarial training module for robust feature learning"""
    def __init__(self, base_model):
        self.base_model = base_model
        self.discriminator = self._build_discriminator()
        
    def _build_discriminator(self):
        """Build discriminator for adversarial training"""
        inputs = layers.Input(shape=(512,))  # Features from base model
        
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Binary classification: real vs fake features
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='discriminator')
    
    def get_feature_extractor(self):
        """Get feature extractor from base model"""
        # Extract features before final classification layer
        feature_layer = self.base_model.get_layer('dense_2')  # Adjust layer name as needed
        return Model(self.base_model.input, feature_layer.output)
