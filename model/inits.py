import tensorflow as tf
import numpy as np

def uniform(shape, scale=0.05, name=None):
    """
    Initializes weights with values drawn from a uniform distribution.

    Args:
        shape (list or tuple): Shape of the weight tensor.
        scale (float): Scaling factor for the uniform distribution.
        name (str): Name of the variable.

    Returns:
        tf.Variable: Variable initialized with values drawn from a uniform distribution.
    """
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    """
    Glorot & Bengio (AISTATS 2010) weight initialization.

    Args:
        shape (list or tuple): Shape of the weight tensor.
        name (str): Name of the variable.

    Returns:
        tf.Variable: Variable initialized with Glorot & Bengio initialization.
    """
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """
    Initializes weights with zeros.

    Args:
        shape (list or tuple): Shape of the weight tensor.
        name (str): Name of the variable.

    Returns:
        tf.Variable: Variable initialized with zeros.
    """
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def ones(shape, name=None):
    """
    Initializes weights with ones.

    Args:
        shape (list or tuple): Shape of the weight tensor.
        name (str): Name of the variable.

    Returns:
        tf.Variable: Variable initialized with ones.
    """
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)
