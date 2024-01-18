import tensorflow as tf
from .layers import Layer, Dense
from .inits import glorot, zeros
from typing import Tuple

class Aggregator(Layer):
    """
    Base class for different aggregation methods.

    Attributes:
        input_dim (int): Dimensionality of the input vectors.
        output_dim (int): Dimensionality of the output vectors after aggregation.
        neigh_input_dim (int): Dimensionality of the neighbor input vectors.
        dropout (float): Dropout rate for regularization during training.
        bias (bool): Whether to include bias terms in the aggregation.
        act (function): Activation function applied to the aggregated output.
        concat (bool): Whether to concatenate the self and neighbor vectors.

    Methods:
        __init__: Initializes the Aggregator with specified parameters.
        _call: Placeholder method for aggregation. Subclasses should implement this method.
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
                 dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        """
        Initializes the Aggregator with specified parameters.

        Args:
            input_dim (int): Dimensionality of the input vectors.
            output_dim (int): Dimensionality of the output vectors after aggregation.
            neigh_input_dim (int): Dimensionality of the neighbor input vectors.
            dropout (float): Dropout rate for regularization during training.
            bias (bool): Whether to include bias terms in the aggregation.
            act (function): Activation function applied to the aggregated output.
            name (str): Name of the Aggregator.
            concat (bool): Whether to concatenate the self and neighbor vectors.
            **kwargs: Additional keyword arguments for the Layer superclass.
        """
        super(Aggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

    def _call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Placeholder for the _call method. Subclasses should implement this method.

        Args:
            inputs (Tuple[tf.Tensor, tf.Tensor]): Tuple containing self and neighbor vectors.

        Returns:
            tf.Tensor: Output tensor after aggregation.
        """
        raise NotImplementedError("Subclasses should implement _call method.")

class MeanAggregator(Aggregator):
    """
    Aggregates via mean followed by matmul and non-linearity.

    Attributes:
        neigh_weights (tf.Variable): Weights for neighbor vectors.
        self_weights (tf.Variable): Weights for self vector.

    Methods:
        __init__: Initializes the MeanAggregator with specified parameters.
        _call: Aggregates using mean and matmul, applying non-linearity.

    Inherits:
        Aggregator: Base class for different aggregation methods.
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
                 dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        """
        Initializes the MeanAggregator with specified parameters.

        Args:
            input_dim (int): Dimensionality of the input vectors.
            output_dim (int): Dimensionality of the output vectors after aggregation.
            neigh_input_dim (int): Dimensionality of the neighbor input vectors.
            dropout (float): Dropout rate for regularization during training.
            bias (bool): Whether to include bias terms in the aggregation.
            act (function): Activation function applied to the aggregated output.
            name (str): Name of the MeanAggregator.
            concat (bool): Whether to concatenate the self and neighbor vectors.
            **kwargs: Additional keyword arguments for the Aggregator superclass.
        """
        super(MeanAggregator, self).__init__(input_dim, output_dim, neigh_input_dim,
                                             dropout, bias, act, name, concat, **kwargs)

        with tf.variable_scope(self.name + '/mean_aggregator_vars'):
            self.neigh_weights = tf.Variable(glorot([self.neigh_input_dim, self.output_dim]),
                                             name='neigh_weights')
            self.self_weights = tf.Variable(glorot([self.input_dim, self.output_dim]),
                                             name='self_weights')
            if self.bias:
                self.bias = tf.Variable(zeros([self.output_dim]), name='bias')

    def _call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Aggregates using mean and matmul, applying non-linearity.

        Args:
            inputs (Tuple[tf.Tensor, tf.Tensor]): Tuple containing self and neighbor vectors.

        Returns:
            tf.Tensor: Output tensor after aggregation.
        """
        self_vecs, neigh_vecs = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs, 1 - self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1 - self.dropout)
        neigh_means = tf.reduce_mean(neigh_vecs, axis=1)

        from_neighs = tf.matmul(neigh_means, self.neigh_weights)
        from_self = tf.matmul(self_vecs, self.self_weights)

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # Add bias if applicable
        if self.bias:
            output += self.bias

        return self.act(output)


class GCNAggregator(Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    Same matmul parameters are used for self vector and neighbor vectors.

    Attributes:
        weights (tf.Variable): Weights for neighbor vectors.
        bias (tf.Variable): Bias term if applicable.

    Methods:
        __init__: Initializes the GCNAggregator with specified parameters.
        _call: Aggregates using mean, matmul, and non-linearity.

    Inherits:
        Layer: Base class for neural network layers.
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
                 dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        """
        Initializes the GCNAggregator with specified parameters.

        Args:
            input_dim (int): Dimensionality of the input vectors.
            output_dim (int): Dimensionality of the output vectors after aggregation.
            neigh_input_dim (int): Dimensionality of the neighbor input vectors.
            dropout (float): Dropout rate for regularization during training.
            bias (bool): Whether to include bias terms in the aggregation.
            act (function): Activation function applied to the aggregated output.
            name (str): Name of the GCNAggregator.
            concat (bool): Whether to concatenate the self and neighbor vectors.
            **kwargs: Additional keyword arguments for the Layer superclass.
        """
        super(GCNAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.variable_scope(self.name + name + '_vars'):
            self.weights = tf.Variable(glorot([neigh_input_dim, output_dim]),
                                       name='neigh_weights')
            if self.bias:
                self.bias = tf.Variable(zeros([output_dim]), name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Aggregates using mean, matmul, and non-linearity.

        Args:
            inputs (Tuple[tf.Tensor, tf.Tensor]): Tuple containing self and neighbor vectors.

        Returns:
            tf.Tensor: Output tensor after aggregation.
        """
        self_vecs, neigh_vecs = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs, 1 - self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1 - self.dropout)

        # Calculate means using both self vector and neighbor vectors
        means = tf.reduce_mean(tf.concat([neigh_vecs, tf.expand_dims(self_vecs, axis=1)], axis=1), axis=1)

        # [nodes] x [out_dim]
        output = tf.matmul(means, self.weights)

        # Add bias if applicable
        if self.bias:
            output += self.bias

        return self.act(output)


class AttentionAggregator(Layer):
    """
    Aggregates using attention mechanism followed by matmul and non-linearity.

    Attributes:
        weights (tf.Variable): Weights for neighbor vectors.
        bias (tf.Variable): Bias term if applicable.

    Methods:
        __init__: Initializes the AttentionAggregator with specified parameters.
        _call: Aggregates using attention mechanism, matmul, and non-linearity.

    Inherits:
        Layer: Base class for neural network layers.
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
                 dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        """
        Initializes the AttentionAggregator with specified parameters.

        Args:
            input_dim (int): Dimensionality of the input vectors.
            output_dim (int): Dimensionality of the output vectors after aggregation.
            neigh_input_dim (int): Dimensionality of the neighbor input vectors.
            dropout (float): Dropout rate for regularization during training.
            bias (bool): Whether to include bias terms in the aggregation.
            act (function): Activation function applied to the aggregated output.
            name (str): Name of the AttentionAggregator.
            concat (bool): Whether to concatenate the self and neighbor vectors.
            **kwargs: Additional keyword arguments for the Layer superclass.
        """
        super(AttentionAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.variable_scope(self.name + name + '_vars'):
            self.weights = tf.Variable(glorot([neigh_input_dim, output_dim]),
                                       name='neigh_weights')
            if self.bias:
                self.bias = tf.Variable(zeros([output_dim]), name='neigh_bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Aggregates using attention mechanism, matmul, and non-linearity.

        Args:
            inputs (Tuple[tf.Tensor, tf.Tensor]): Tuple containing self and neighbor vectors.

        Returns:
            tf.Tensor: Output tensor after aggregation.
        """
        self_vecs, neigh_vecs = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs, 1 - self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1 - self.dropout)

        # Reshape from [batch_size, depth] to [batch_size, 1, depth] for matmul.
        query = tf.expand_dims(self_vecs, 1)
        neigh_self_vecs = tf.concat([neigh_vecs, query], axis=1)

        # Calculate attention scores using matmul and softmax
        score = tf.matmul(query, neigh_self_vecs, transpose_b=True)
        score = tf.nn.softmax(score, axis=-1)

        # Align based on attention scores
        context = tf.matmul(score, neigh_self_vecs)
        context = tf.squeeze(context, [1])

        # [nodes] x [out_dim]
        output = tf.matmul(context, self.weights)

        # Add bias if applicable
        if self.bias:
            output += self.bias

        return self.act(output)


class MaxPoolingAggregator(Layer):
    """
    Aggregates via max-pooling over MLP functions.

    Attributes:
        dropout (float): Dropout rate for regularization during training.
        bias (bool): Whether to include bias terms in the aggregation.
        act (function): Activation function applied to the aggregated output.
        concat (bool): Whether to concatenate the self and neighbor vectors.
        hidden_dim (int): Dimensionality of hidden layer in MLP processing.

    Methods:
        __init__: Initializes the MaxPoolingAggregator with specified parameters.
        _call: Aggregates using max-pooling, MLP processing, and non-linearity.

    Inherits:
        Layer: Base class for neural network layers.
    """

    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
                 dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        """
        Initializes the MaxPoolingAggregator with specified parameters.

        Args:
            input_dim (int): Dimensionality of the input vectors.
            output_dim (int): Dimensionality of the output vectors after aggregation.
            model_size (str): Size of the model, either "small" or "big".
            neigh_input_dim (int): Dimensionality of the neighbor input vectors.
            dropout (float): Dropout rate for regularization during training.
            bias (bool): Whether to include bias terms in the aggregation.
            act (function): Activation function applied to the aggregated output.
            name (str): Name of the MaxPoolingAggregator.
            concat (bool): Whether to concatenate the self and neighbor vectors.
            **kwargs: Additional keyword arguments for the Layer superclass.
        """
        super(MaxPoolingAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if model_size == "small":
            hidden_dim = self.hidden_dim = 512
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 1024

        # MLP layers for processing neighbor vectors
        self.mlp_layers = []
        self.mlp_layers.append(Dense(input_dim=neigh_input_dim,
                                     output_dim=hidden_dim,
                                     act=tf.nn.relu,
                                     dropout=dropout,
                                     sparse_inputs=False,
                                     logging=self.logging))

        with tf.variable_scope(self.name + name + '_vars'):
            # Weights for neighbor vectors and self vector
            self.vars['neigh_weights'] = glorot([hidden_dim, output_dim],
                                                name='neigh_weights')
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                               name='self_weights')

            # Bias if applicable
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

    def _call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Aggregates using max-pooling, MLP processing, and non-linearity.

        Args:
            inputs (Tuple[tf.Tensor, tf.Tensor]): Tuple containing self and neighbor vectors.

        Returns:
            tf.Tensor: Output tensor after aggregation.
        """
        self_vecs, neigh_vecs = inputs
        neigh_h = neigh_vecs

        dims = tf.shape(neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]

        # Reshape neighbor vectors for MLP processing
        h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))

        # Apply MLP layers to neighbor vectors
        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)

        # Reshape back to original shape after MLP processing
        neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim))

        # Max-pooling over the neighbor vectors
        neigh_h = tf.reduce_max(neigh_h, axis=1)

        # Apply weights to neighbor and self vectors
        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])

        # Aggregate results, either by addition or concatenation
        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # Add bias if applicable
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class MeanPoolingAggregator(Layer):
    """
    Aggregates via mean-pooling over MLP functions.

    Attributes:
        dropout (float): Dropout rate for regularization during training.
        bias (bool): Whether to include bias terms in the aggregation.
        act (function): Activation function applied to the aggregated output.
        concat (bool): Whether to concatenate the self and neighbor vectors.
        hidden_dim (int): Dimensionality of hidden layer in MLP processing.

    Methods:
        __init__: Initializes the MeanPoolingAggregator with specified parameters.
        _call: Aggregates using mean-pooling, MLP processing, and non-linearity.

    Inherits:
        Layer: Base class for neural network layers.
    """

    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
                 dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        """
        Initializes the MeanPoolingAggregator with specified parameters.

        Args:
            input_dim (int): Dimensionality of the input vectors.
            output_dim (int): Dimensionality of the output vectors after aggregation.
            model_size (str): Size of the model, either "small" or "big".
            neigh_input_dim (int): Dimensionality of the neighbor input vectors.
            dropout (float): Dropout rate for regularization during training.
            bias (bool): Whether to include bias terms in the aggregation.
            act (function): Activation function applied to the aggregated output.
            name (str): Name of the MeanPoolingAggregator.
            concat (bool): Whether to concatenate the self and neighbor vectors.
            **kwargs: Additional keyword arguments for the Layer superclass.
        """
        super(MeanPoolingAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if model_size == "small":
            hidden_dim = self.hidden_dim = 512
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 1024

        # MLP layers for processing neighbor vectors
        self.mlp_layers = []
        self.mlp_layers.append(Dense(input_dim=neigh_input_dim,
                                     output_dim=hidden_dim,
                                     act=tf.nn.relu,
                                     dropout=dropout,
                                     sparse_inputs=False,
                                     logging=self.logging))

        with tf.variable_scope(self.name + name + '_vars'):
            # Weights for neighbor vectors and self vector
            self.vars['neigh_weights'] = glorot([hidden_dim, output_dim],
                                                name='neigh_weights')
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                               name='self_weights')

            # Bias if applicable
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

    def _call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Aggregates using mean-pooling, MLP processing, and non-linearity.

        Args:
            inputs (Tuple[tf.Tensor, tf.Tensor]): Tuple containing self and neighbor vectors.

        Returns:
            tf.Tensor: Output tensor after aggregation.
        """
        self_vecs, neigh_vecs = inputs
        neigh_h = neigh_vecs

        dims = tf.shape(neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]

        # Reshape neighbor vectors for MLP processing
        h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))

        # Apply MLP layers to neighbor vectors
        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)

        # Reshape back to original shape after MLP processing
        neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim))

        # Mean-pooling over the neighbor vectors
        neigh_h = tf.reduce_mean(neigh_h, axis=1)

        # Apply weights to neighbor and self vectors
        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])

        # Aggregate results, either by addition or concatenation
        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # Add bias if applicable
        if self.bias:
            output += self.vars['bias']

        return self.act(output)
