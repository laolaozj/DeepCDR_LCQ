import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input,InputLayer,Multiply,ZeroPadding2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Conv1D,MaxPooling1D
from tensorflow.keras.layers import Dense,Activation,Dropout,Flatten,Concatenate
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dropout,GlobalMaxPooling1D,GlobalAveragePooling1D,GlobalMaxPooling2D,GlobalAveragePooling2D
from tensorflow.keras.models import Model
from layers.graph import GraphConvTest
import tensorflow as tf
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

class KerasMultiSourceGCNModel(object):

    def __init__(self,use_mut,use_gexp,use_methy,regr=True):#
        self.use_mut = use_mut
        self.use_gexp = use_gexp
        self.use_methy = use_methy
        self.regr = regr

    def createMaster(self,drug_dim,edge_dim,mutation_dim,gexpr_dim,methy_dim,units_list,unit_edge_list,batch,dropout,activation,use_relu=True,use_bn=True,use_GMP=True):

        node_feature = Input(batch_shape=(batch,100,drug_dim),name='node_feature')#drug_dim=33 (1,100,33)
        lcq_adj = Input(batch_shape=(batch,100,100,edge_dim),name='lcq_adj') #(1,100,100,11)
        # edge_feature=Input(shape=(None,edge_dim),name='edge_feature')
        mutation_input = Input(batch_shape=(batch,1,mutation_dim,1),name='mutation_feat_input') #(1, 1, 34673, 1)
        gexpr_input = Input(batch_shape=(batch,gexpr_dim,),name='gexpr_feat_input') #(1, 697)
        methy_input = Input(batch_shape=(batch,methy_dim,),name='methy_feat_input') #(1, 808)

        #drug feature with GCN

        GCN_layer = GraphConvTest(units=units_list[0],units_edge=unit_edge_list[0],step=0)([node_feature,lcq_adj])

        # if use_relu:
        #     # GCN_layer = [Activation('relu')(item) for item in GCN_layer]
        #     GCN_layer = [Activation('relu')(item) for item in GCN_layer]
        # else:
        #     GCN_layer = [Activation('tanh')(item) for item in GCN_layer]
        GCN_layer = [Activation(activation)(item) for item in GCN_layer]

        if use_bn:
            GCN_layer = [BatchNormalization()(item) for item in GCN_layer]
        GCN_layer = [Dropout(dropout)(item) for item in GCN_layer]

        # for i in range(2):
        #     GCN_layer = GraphConvTest(units=units_list[i+1],units_edge=unit_edge_list[i+1],step=i+1)(GCN_layer)
        #     if use_relu:
        #         GCN_layer = [Activation('relu')(item) for item in GCN_layer]
        #     else:
        #         GCN_layer = [Activation('tanh')(item) for item in GCN_layer]
        #     if use_bn:
        #         GCN_layer = [BatchNormalization()(item) for item in GCN_layer]
        #     GCN_layer = [Dropout(0.1)(item) for item in GCN_layer]
        #last layer, do not update edge as it will introduce unused trainable weights.
        GCN_layer = GraphConvTest(units=units_list[-1],units_edge=unit_edge_list[-1],step=1 ,update_edge=False)(GCN_layer)

        # if use_relu:
        #     GCN_layer = [Activation('relu')(item) for item in GCN_layer]
        # else:
        #     GCN_layer = [Activation('tanh')(item) for item in GCN_layer]
        GCN_layer = [Activation(activation)(item) for item in GCN_layer]

        if use_bn:
            GCN_layer = [BatchNormalization()(item) for item in GCN_layer]
        GCN_layer = [Dropout(dropout)(item) for item in GCN_layer]

        #global pooling
        if use_GMP:
            x_drug = GlobalMaxPooling1D()(GCN_layer[0])
            x_drug_edge = GlobalMaxPooling2D()(GCN_layer[1])
        else:
            x_drug = GlobalAveragePooling1D()(GCN_layer[0])
            x_drug_edge = GlobalAveragePooling2D()(GCN_layer[1])
        print(x_drug.shape,"wulawula")
        # x_drug= merge([x_drug,x_drug_edge],mode='concat')
        x_drug= Concatenate()([x_drug, x_drug_edge])
        print(x_drug.shape,"wulawula")#128+11

        #genomic mutation feature
        x_mut = Conv2D(filters=50, kernel_size=(1,700),strides=(1, 5), activation = 'tanh',padding='valid')(mutation_input)
        x_mut = MaxPooling2D(pool_size=(1,5))(x_mut)
        x_mut = Conv2D(filters=30, kernel_size=(1,5),strides=(1, 2), activation = 'relu',padding='valid')(x_mut)
        x_mut = MaxPooling2D(pool_size=(1,10))(x_mut)
        x_mut = Flatten()(x_mut)
        x_mut = Dense(100,activation = 'relu')(x_mut)
        x_mut = Dropout(0.1)(x_mut)
        #gexp feature
        x_gexpr = Dense(256)(gexpr_input)
        x_gexpr = Activation('tanh')(x_gexpr)
        x_gexpr = BatchNormalization()(x_gexpr)
        x_gexpr = Dropout(0.1)(x_gexpr)
        x_gexpr = Dense(100,activation='relu')(x_gexpr)
        #methylation feature
        x_methy = Dense(256)(methy_input)
        x_methy = Activation('tanh')(x_methy)
        x_methy = BatchNormalization()(x_methy)
        x_methy = Dropout(0.1)(x_methy)
        x_methy = Dense(100,activation='relu')(x_methy)
        x = x_drug

        if self.use_mut:
            x = Concatenate(axis=1)([x,x_mut])
        if self.use_gexp:
            x = Concatenate()([x,x_gexpr])
        if self.use_methy:
            x = Concatenate()([x,x_methy])
        x = Concatenate()([x_mut,x_drug,x_gexpr,x_methy])
        x = Dense(300,activation = 'tanh')(x)
        x = Dropout(0.1)(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=-1))(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=1))(x)
        x = Conv2D(filters=30, kernel_size=(1,150),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,2))(x)
        x = Conv2D(filters=10, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,3))(x)
        x = Conv2D(filters=5, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,3))(x)
        x = Dropout(0.1)(x)
        x = Flatten()(x)
        x = Dropout(0.2)(x)
        if self.regr:
            output = Dense(1,name='output')(x)
        else:
            output = Dense(1,activation = 'sigmoid',name='output')(x)

        model = Model(inputs=[node_feature,lcq_adj,mutation_input,gexpr_input,methy_input],outputs=output)

        return model

