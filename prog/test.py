import numpy as np
import tensorflow as tf

#%%

from layers_transformer import ffn_layer, attention_layer, prepost_layer
from layers_transformer import embedding_layer, position_embedding


class TransformerModule(tf.keras.layers.Layer):
    """Transformer module.
    The module is made up of N identical layers. Each layer is composed
    of the sublayers:
        1. Self-attention layer
        2. Feedforward network (2 fully-connected layers)
    """

    def __init__(self, params):
        super(TransformerModule, self).__init__()
        self.params = params
        self.layers = []

    def build(self, input_shape):
        """Builds the Transformer module."""
        params = self.params
        for _ in range(params["num_hidden_layers"]):
            # Create sublayers for each layer.
            self_attention_layer = attention_layer.SelfAttention(
                params["hidden_size"], params["num_heads"],
                params["attention_dropout"])
            feed_forward_network = ffn_layer.FeedForwardNetwork(
                params["hidden_size"], params["filter_size"], params["relu_dropout"])

            self.layers.append([
                prepost_layer.PrePostProcessingAttWrapper(self_attention_layer, params),
                prepost_layer.PrePostProcessingFnnWrapper(feed_forward_network, params)
            ])
        # Create final layer normalization layer.
        self.output_normalization = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, dtype="float32")
        super(TransformerModule, self).build(input_shape)

    def get_config(self):
        return {
            "params": self.params,
        }

    def call(self, inputs, attention_bias, training):
        """Return the output of the transformer module.
        Args:
            inputs: tensor with shape [batch_size, input_length, hidden_size]
            attention_bias: bias for the encoder self-attention layer. [batch_size, 1,
                1, input_length]
            training: boolean, whether in training mode or not.
        Returns:
            Outputs of transformer module.
                item[0]: transformer encoded results
                        float32 tensor with shape [batch_size, input_length, hidden_size]
                item[1]: self-attention weights
                        float32 tensor with shape [batch_size, num_hidden_layers, num_heads, input_length, input_length]
        """

        attention_weights = {}
        x = inputs
        for i, layer in enumerate(self.layers):
            self.params["current_layer"] = i
            # Run inputs through the sublayers.
            self_attention_layer = layer[0]
            feed_forward_network = layer[1]

            with tf.name_scope("layer_%d" % i):
                with tf.name_scope("self_attention"):
                    x, w = self_attention_layer(
                        x, attention_bias, training=training)
                    attention_weights['layer_%d' % i] = w
                with tf.name_scope("ffn"):
                    x = feed_forward_network(
                        x, training=training)

        return self.output_normalization(x), attention_weights


# %% md

## Create a model instance

# %%

params = {
    "num_hidden_layers": 8,
    "hidden_size": 256,
    "num_heads": 8,
    "attention_dropout": 0.1,
    "filter_size": 128,
    "relu_dropout": 0.1,
    "layer_postprocess_dropout": 0.1,
    "dtype": "float32"
}

# %%

layer = TransformerModule(params)

# %% md

## Define input here

# %%

# define input here
nb_genes = 697
nb_feats = 4
batch_size = 16
inputs = tf.zeros(shape=(batch_size, nb_genes, nb_feats))

# %%

# add position embedding
position_embedding_layer = position_embedding.RelativePositionEmbedding(
    hidden_size=nb_feats)
pos_encoding = position_embedding_layer(inputs=inputs)
pos_encoding = tf.cast(pos_encoding, params["dtype"])
# pos_encoding with shape [input_length, hidden_size], broadcast here.
inputs = inputs+pos_encoding

# %%

attention_bias = tf.zeros_like(tf.reduce_mean(inputs, axis=-1))
attention_bias = tf.expand_dims(
    tf.expand_dims(attention_bias, axis=1), axis=1)

# %%

print(inputs.shape, attention_bias.shape)
output = layer(inputs, attention_bias, training=True)

#%%

#output contains two part, currently, we only need the first part
print(output[0].shape)
