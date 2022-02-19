from __future__ import print_function

from keras import activations, initializers, constraints
from keras import regularizers
from keras.engine import Layer
import keras.backend as K


import keras

class GraphLayer(keras.layers.Layer):

    def __init__(self,
                 in_features,
                 out_features,
                 step_num=1,
                 activation="tanh",
                 **kwargs):
        """Initialize the layer.

        :param step_num: Two nodes are considered as connected if they could be reached in `step_num` steps.
        :param activation: The activation function after convolution.
        :param kwargs: Other arguments for parent class.
        """
        super(GraphLayer, self).__init__(**kwargs)

        self.in_features=in_features
        self.out_features=out_features
        self.bn=bn
        self.tanh=K.tanh()
        self.relu = K.relu()

        self.weight=(self.in_features, self.out_features)
        self.bias =(self.out_features)
        self.supports_masking = True
        self.step_num = step_num
        self.reset_parameters()

        if bn:
            self.mol_bn1 = keras.layers.BatchNormalization(self.num_head * self.out_features)
            self.mol_bn2 = keras.layers.BatchNormalization(self.out_features)
            self.adj_bn1 = keras.layers.BatchNormalization(16)
            self.adj_bn2 = keras.layers.BatchNormalization(self.num_head)
        if sn:
            self.out_linear=SN(nn.Linear(self.out_features*self.num_head, self.out_features))
        else:
            self.out_linear=(nn.Linear(self.out_features*self.num_head, self.out_features, bias=True))

        self.activation = keras.activations.get(activation)
        self.supports_masking = True


    def get_config(self):
        config = {
            'step_num': self.step_num,
            'activation': self.activation,
        }
        base_config = super(GraphLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, **kwargs):
        features, edges = inputs
        edges = K.cast(edges, K.floatx())
        if self.step_num > 1:
            edges = self._get_walked_edges(edges, self.step_num)
        outputs = self.activation(self._call(features, edges))
        return outputs


class GraphConv(GraphLayer):
    """Graph convolutional layer.

    h_i^{(t)} = \sigma \left ( \frac{ G_i^T (h_i^{(t - 1)} W + b)}{\sum G_i}  \right )
    """

    def __init__(self,
                 units,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 use_bias=True,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 **kwargs):
        """Initialize the layer.

        :param units: Number of new states. If the input shape is (batch_size, node_num, feature_len), then the output
                      shape is (batch_size, node_num, units).
        :param kernel_initializer: The initializer of the kernel weight matrix.
        :param kernel_regularizer: The regularizer of the kernel weight matrix.
        :param kernel_constraint:  The constraint of the kernel weight matrix.
        :param use_bias: Whether to use bias term.
        :param bias_initializer: The initializer of the bias vector.
        :param bias_regularizer: The regularizer of the bias vector.
        :param bias_constraint: The constraint of the bias vector.
        :param kwargs: Other arguments for parent class.
        """
        self.units = units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.use_bias = use_bias
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.bias_constraint = keras.constraints.get(bias_constraint)



        # self.W, self.b = None, None
        super(GraphConv, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'units': self.units,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'use_bias': self.use_bias,
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
        }
        base_config = super(GraphConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        feature_dim = input_shape[0][2]
        self.W = self.add_weight(
            shape=(feature_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='{}_W'.format(self.name),
        )
        if self.use_bias:
            self.b = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='{}_b'.format(self.name),
            )
        super(GraphConv, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + (self.units,)

    def call(self, inputs, **kwargs):
        node_feature, lcq_adj,
        ,mask,adjmask = inputs

        a0, a1, a2, a3 = edge_feature.shape  # batch*channel*sha1*sha2 edj 11
        m1, m2, m3 = node_feature.shape  # batch*sha1*emb   node     35
        assert a2 == a3

        support = K.dot(node_feature.reshape(m1 * m2, m3), self.weight).reshape(m1, m2, self.out_features)
        output = K.dot(adj.reshape(a0, a1 * a2, a3), support).reshape(a0, a1, a2, self.out_features)+support.unsqueeze(
            1)
        assert ((output * (1-mask.unsqueeze(1))).sum() == 0)
        if self.bias is not None:
            output = output+self.bias
        output = self.activation(output.permute(0, 2, 3, 1).reshape(m1, m2, m3 * self.num_head))
        if self.bn:
            output = self.mol_bn2(output.reshape(-1, self.out_features)).reshape(m1, m2, self.out_features)
        output = output * mask
        output = self.out_linear(output)
        output = self.activation(output)
        if self.bn:
            output = self.mol_bn2(output.reshape(-1, self.out_features)).reshape(m1, m2, self.out_features)
        output = output * mask

        tadj = K.concatenate([output.unsqueeze(1) * K.ones([1, a2, 1, 1]).cuda(output.device),
                              output.unsqueeze(2) * K.ones([1, 1, a3, 1]).cuda(output.device)], 3)
        tadj = self.relu(self.adj_linear(tadj))
        if self.bn:
            tadj = self.adj_bn1(tadj.reshape(-1, 16)).reshape(a0, a2, a3, 16)
        tadj = K.concatenate([tadj, adj.permute(0, 2, 3, 1)], 3)
        _adj = adj
        adj = self.relu(self.adj_out_linear(tadj))
        if self.bn:
            adj = self.adj_bn2(adj.reshape(-1, self.num_head)).reshape(a0, a2, a3, self.num_head)
        adj = adj * adjmask
        adj = adj.permute(0, 3, 1, 2)+_adj
        return [output, adj, mask, adjmask]

