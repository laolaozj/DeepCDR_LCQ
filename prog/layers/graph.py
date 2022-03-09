from __future__ import print_function

from keras import activations, initializers, constraints
from keras import regularizers
from keras.engine import Layer
from  keras.backend.tensorflow_backend import clip
import tensorflow as tf
from keras import backend as K
import numpy as np
from keras.layers import  Reshape
import keras
import tensorflow as tf
from keras.layers import Dense,Activation,Dropout,Flatten,Concatenate
from keras.layers import BatchNormalization


class GraphLayer(keras.layers.Layer):

    def __init__(self,
                 step_num=1,
                 activation="tanh",
                 **kwargs):
        """Initialize the layer.

        :param step_num: Two nodes are considered as connected if they could be reached in `step_num` steps.
        :param activation: The activation function after convolution.
        :param kwargs: Other arguments for parent class.
        """
        self.supports_masking = True
        self.step_num = step_num
        self.activation = keras.activations.get(activation)
        self.supports_masking = True

        self.out_features = 33
        self.num_head = 11

        super(GraphLayer, self).__init__(**kwargs)

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
                 kernel_constraint=None,
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
        self.test_W = self.add_weight(
            shape=(feature_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='{}_W'.format(self.name),
            trainable=True
        )
        self.adj_linear_W = self.add_weight(
            shape=(self.out_features*2, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='{}_W'.format(self.name),
            trainable=True
        )
        self.adj_out_linear_W = self.add_weight(
            shape=(self.units+self.num_head, self.num_head),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='{}_W'.format(self.name),
            trainable=True
        )
        self.out_linear_W = self.add_weight(
            shape=(feature_dim*self.num_head, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='{}_W'.format(self.name),
            trainable=True
        )
        if self.use_bias:
            self.adj_linear_b = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='{}_b'.format(self.name),
                trainable=True
            )
            self.adj_out_linear_b = self.add_weight(
                shape=(self.num_head,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='{}_b'.format(self.name),
                trainable=True
            )
            self.out_linear_b = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='{}_b'.format(self.name),
                trainable=True
            )

        super(GraphConv, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        print("qweqwead")
        return input_shape

    def call(self, inputs, **kwargs):
        node_feature, lcq_adj=inputs
        lcq_adj=tf.transpose(lcq_adj,(0,3,1,2))
    
        a0, a1, a2, a3 = lcq_adj.shape  # batch*channel*sha1*sha2 edj 32   11 100  100
        m1, m2, m3 = node_feature.shape  # batch*sha1*emb   node     32    100 33

        node_feature_test=node_feature[:,:,0]
        #mask = K.clip(node_feature_test,0.0, 1.0)
        #mask=K.expand_dims(mask,2)
        #adjmask1=K.expand_dims(mask,1)*K.expand_dims(mask,2)
        #test=K.eye(int(m2))
        #adjmask2=K.expand_dims(test,0)
        #adjmask2=K.expand_dims(adjmask2,3)
        #adjmask=adjmask1-adjmask2
        #adjmask = K.clip(adjmask,0, 1)
        #print(adjmask.shape,"adjmask")

        # adjmask = (mask.unsqueeze(1) * mask.unsqueeze(2))-K.eye(a2).unsqueeze(0).unsqueeze(3).cuda(
        #     mol.device)
        # +torch.eye(adj.shape[1]).cuda().unsqueeze(0).unsqueeze(3)*maskmol.unsqueeze(3)
        support=K.reshape(node_feature,(m1*m2,m3))
        support=K.dot(support,self.test_W)
        support=K.reshape(support,(m1,m2,self.out_features))
        # support = K.dot(node_feature.reshape(m1 * m2, m3), self.test_W).reshape(m1, m2, self.out_features)
        output1=K.reshape(lcq_adj,(a0,a1*a2,a3))
        output1=K.batch_dot(output1,support)
        output1=K.reshape(output1,(a0,a1,a2,self.out_features))
        # output1 = K.dot(lcq_adj.reshape(a0, a1 * a2, a3), support).reshape(a0, a1, a2, self.out_features)
        output2=K.expand_dims(support,1)
        output=output1+output2
        print(output.shape,"output1.shape")
        # test=1-K.expand_dims(mask,1)
        # output=output*test
        # res=K.sum(output)
        # assert ((output * (1-mask.unsqueeze(1))).sum() == 0)
        # if self.b is not None:
        #     output = output+self.b
        output=K.permute_dimensions(output,(0,2,3,1))
        print(output.shape,"output2.shape")
        output=K.reshape(output,(m1,m2,m3*self.num_head))
        print(output.shape,"output3.shape")
        output=self.activation(output)
        # output = self.activation(output.permute(0,     2, 3, 1).reshape(m1, m2, m3 * self.num_head))
        # if self.bn:
        #     output = self.mol_bn2(output.reshape(-1, self.out_features)).reshape(m1, m2, self.out_features)
        #output = output * mask
        # output = self.out_linear(output)
        output = K.dot(output, self.out_linear_W)
        if self.use_bias:
            output += self.out_linear_b
        output = self.activation(output)
        # if self.bn:
        #     output = self.mol_bn2(output.reshape(-1, self.out_features)).reshape(m1, m2, self.out_features)
        #output = output * mask
        print(output.shape,"final outputshape")


        tadj = K.concatenate([K.expand_dims(output,1) * K.ones([1, a2, 1, 1]),
                              K.expand_dims(output,2) * K.ones([1, 1, a3, 1])], 3)
        # tadj = self.relu(self.adj_linear(tadj))
        print(tadj.shape, "tadj1")
        tadj = K.dot(tadj, self.adj_linear_W)
        if self.use_bias:
            tadj += self.adj_linear_b
        tadj=K.relu(tadj)
        print(tadj.shape, "tadj2")
        # if self.bn:
        #     tadj = self.adj_bn1(tadj.reshape(-1, 16)).reshape(a0, a2, a3, 16)

        tadj=K.concatenate([tadj,K.permute_dimensions(lcq_adj,(0,2,3,1))],3)
        # tadj = K.concatenate([tadj, lcq_adj.permute(0, 2, 3, 1)], 3)
        print(tadj.shape,"tadj3")
        _adj = lcq_adj

        # adj = self.relu(self.adj_out_linear(tadj))
        adj = K.dot(tadj, self.adj_out_linear_W)
        print(adj.shape,"adj1shape")
        if self.use_bias:
            adj += self.adj_out_linear_b
        adj=K.relu(adj)
        print(adj.shape,"adj2shape")
        #adj = adj * adjmask
        print(adj.shape,"adj3shape")
        adj=K.permute_dimensions(adj,(0,3,1,2,))+_adj
        adj=tf.transpose(adj,(0,2,3,1))

        print(adj.shape,"adjfinalshape")

        act_output = Activation('relu')(output)
        ba_output = BatchNormalization()(act_output)
        dr_output = Dropout(0.1)(ba_output)

        act_adj = Activation('relu')(adj)
        ba_adj = BatchNormalization()(act_adj)
        dr_adj = Dropout(0.1)(ba_adj)

        print(dr_adj, 'out_adj')
        return [dr_output, dr_adj]

class GraphConvTest(keras.layers.Layer):

    def __init__(self,
                 units,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 use_bias=False,
                 update_edge=True,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 **kwargs):
        self.units = units
        self.use_bias = use_bias
        self.activation = keras.activations.get('tanh')
        self.out_features = 33
        self.num_head = 11
        self.update_edge = update_edge
        super(GraphConvTest, self).__init__(**kwargs)


    def build(self, input_shape):
        feature_dim = input_shape[0][2]
        self.test_W = self.add_weight(
            name = 'w1',
            shape=(feature_dim, self.units),
            initializer='random_normal',
            trainable=True
        )
        self.out_linear_W = self.add_weight(
            name = 'w4',
            shape=(feature_dim*self.num_head, self.units),
            initializer='random_normal',
            trainable=True
        )
        if self.update_edge:
            self.adj_linear_W = self.add_weight(
                name = 'w2',
                shape=(self.out_features*2, self.units),
                initializer='random_normal',
                trainable=True
            )
            self.adj_out_linear_W = self.add_weight(
                name = 'w3',
                shape=(self.units+self.num_head, self.num_head),
                initializer='random_normal',
                trainable=True
            )
        if self.use_bias:
            self.out_linear_b = self.add_weight(
                name = 'b3',
                shape=(self.units,),
                initializer='random_normal',
                trainable=True
            )
            if self.update_edge:
                self.adj_linear_b = self.add_weight(
                    name = 'b1',
                    shape=(self.units,),
                    initializer='random_normal',
                    trainable=True
                )
                self.adj_out_linear_b = self.add_weight(
                    name = 'b2',
                    shape=(self.num_head,),
                    initializer='random_normal',
                    trainable=True
                )
        super(GraphConvTest, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        print("qweqwead")
        return input_shape

    def call(self, inputs, **kwargs):
        node_feature, lcq_adj=inputs
        lcq_adj=tf.transpose(lcq_adj,(0,3,1,2))
    
        a0, a1, a2, a3 = lcq_adj.shape  # batch*channel*sha1*sha2 edj 32   11 100  100
        m1, m2, m3 = node_feature.shape  # batch*sha1*emb   node     32    100 33

        support=K.reshape(node_feature,(m1*m2,m3))
        support=K.dot(support,self.test_W)
        support=K.reshape(support,(m1,m2,self.out_features))
        # support = K.dot(node_feature.reshape(m1 * m2, m3), self.test_W).reshape(m1, m2, self.out_features)
        output1=K.reshape(lcq_adj,(a0,a1*a2,a3))
        output1=K.batch_dot(output1,support)
        output1=K.reshape(output1,(a0,a1,a2,self.out_features))
        # output1 = K.dot(lcq_adj.reshape(a0, a1 * a2, a3), support).reshape(a0, a1, a2, self.out_features)
        output2=K.expand_dims(support,1)
        output=output1+output2
        print(output.shape,"output1.shape") #(1, 11, 100, 33)
        # test=1-K.expand_dims(mask,1)
        # output=output*test
        # res=K.sum(output)
        # assert ((output * (1-mask.unsqueeze(1))).sum() == 0)
        # if self.b is not None:
        #     output = output+self.b
        output=K.permute_dimensions(output,(0,2,3,1))
        print(output.shape,"output2.shape") #(1, 100, 33, 11)
        output=K.reshape(output,(m1,m2,m3*self.num_head))
        print(output.shape,"output3.shape") #(1, 100, 363)
        output=self.activation(output) 

        # output = self.activation(output.permute(0,     2, 3, 1).reshape(m1, m2, m3 * self.num_head))
        # if self.bn:
        #     output = self.mol_bn2(output.reshape(-1, self.out_features)).reshape(m1, m2, self.out_features)
        #output = output * mask
        # output = self.out_linear(output)
        output = K.dot(output, self.out_linear_W)
        if self.use_bias:
            output += self.out_linear_b
        output = self.activation(output)
        # if self.bn:
        #     output = self.mol_bn2(output.reshape(-1, self.out_features)).reshape(m1, m2, self.out_features)
        #output = output * mask
        print(output.shape,"final outputshape")

        if self.update_edge:
            tadj = K.concatenate([K.expand_dims(output,1) * K.ones([1, a2, 1, 1]),
                                K.expand_dims(output,2) * K.ones([1, 1, a3, 1])], 3)
            # tadj = self.relu(self.adj_linear(tadj))
            print(tadj.shape, "tadj1") #(1, 100, 100, 66)
            tadj = K.dot(tadj, self.adj_linear_W)
            if self.use_bias:
                tadj += self.adj_linear_b
            tadj=K.relu(tadj)
            print(tadj.shape, "tadj2") #(1, 100, 100, 33)

            tadj=K.concatenate([tadj,K.permute_dimensions(lcq_adj,(0,2,3,1))],3)
            # tadj = K.concatenate([tadj, lcq_adj.permute(0, 2, 3, 1)], 3)
            print(tadj.shape,"tadj3")
            _adj = lcq_adj

            # adj = self.relu(self.adj_out_linear(tadj))
            adj = K.dot(tadj, self.adj_out_linear_W)
            print(adj.shape,"adj1shape")
            if self.use_bias:
                adj += self.adj_out_linear_b
            adj=K.relu(adj)
            print(adj.shape,"adj2shape")
            #adj = adj * adjmask
            print(adj.shape,"adj3shape")
            adj=K.permute_dimensions(adj,(0,3,1,2,))+_adj
            adj=tf.transpose(adj,(0,2,3,1))

            print(adj.shape,"adjfinalshape")
            return [output, adj]
        else:
            lcq_adj=tf.transpose(lcq_adj,(0,2,3,1))
            return [output, lcq_adj]