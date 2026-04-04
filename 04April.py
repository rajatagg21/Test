import tensorflow as tf
from keras_nlp.models import RetVec


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads
        )
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.norm1 = tf.keras.layers.LayerNormalization()

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.norm2 = tf.keras.layers.LayerNormalization()

    def call(self, x, training=False):
        # Self-attention
        attn_output = self.attn(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        x = self.norm1(x + attn_output)

        # Feed-forward
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output, training=training)
        x = self.norm2(x + ffn_output)

        return x


class RetVecTransformerClassifier(tf.keras.Model):
    def __init__(
        self,
        num_classes,
        sequence_length=256,
        embed_dim=256,
        num_heads=4,
        ff_dim=512,
        num_attn_layers=1,   # 👈 NEW
        dropout=0.1,
        retvec_trainable=False,
    ):
        super().__init__()

        # ---- RetVec ----
        self.retvec = RetVec(
            sequence_length=sequence_length,
            trainable=retvec_trainable
        )

        # ---- Positional Embedding ----
        self.pos_embedding = tf.keras.layers.Embedding(
            input_dim=sequence_length,
            output_dim=embed_dim
        )

        # ---- Stack of Transformer Blocks ----
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_attn_layers)
        ]

        # ---- Classifier ----
        self.pool = tf.keras.layers.GlobalAveragePooling1D()
        self.classifier = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training=False):
        # ---- RetVec ----
        x = self.retvec(inputs)  # (B, seq_len, embed_dim)

        # ---- Positional Encoding ----
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        pos_embed = self.pos_embedding(positions)
        x = x + pos_embed

        # ---- Transformer stack ----
        for block in self.transformer_blocks:
            x = block(x, training=training)

        # ---- Pool + classify ----
        x = self.pool(x)
        logits = self.classifier(x)

        return logits
