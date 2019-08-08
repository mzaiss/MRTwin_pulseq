import tensorflow as tf
import numpy as np
from tensorflow.python.platform import tf_logging as logging
import numbers
from tensorflow.python.framework import ops
from tensorflow.python.ops import standard_ops

def tf_hard_sigmoid(u):
    h = tf.minimum(1.0, tf.maximum(0.0, u))
    return h

def tf_stretch(s, interval):
    """
    s: 1-d a tensor
    interval: [a,b], the lower/upper bound of the interval to stretch up
    """
    s_bar = s * (interval[1] - interval[0]) + interval[0]
    return s_bar

def tf_concrete_transoform(u, alpha, beta):
    '''
    :param u: a random variable, usually from uniform distribution
    :param alpha:
    :param beta:
    :return:
    '''
    ct = tf.nn.sigmoid((tf.log(u) - tf.log(1 - u) + tf.log(alpha)) / beta)
    return ct


def l0_computation(tensor,
                   interval=[-0.1, 1.1],
                   mu_c=np.log(3 / 7),
                   sigma_c=1e-3,
                   beta=2/3, seed=None):

    # tensor-shape check:
    tensor_shape = tensor.get_shape().as_list() # use the list format
    if len(tensor_shape) > 2:
        raise ValueError("The tensor_shape with rank not larger than 2 is not supported")
    elif None in tensor_shape:
        tensor_shape = [1 if d is None else d for d in tensor_shape] # for input format [?,d] or [d,?], designed for l0_layer application

    u = tf.random_uniform(shape=tensor_shape, dtype=tf.float32, seed=seed)

    c = tf.Variable(tf.random_normal(shape=tensor_shape, mean=mu_c, stddev=sigma_c, seed=seed),
                    name='l0_mask',
                    #collections=['l0_vars'], # todo: the uninitializable issue to be discussed
                    dtype=tf.float32)

    alpha = tf.exp(c)
    s = tf_concrete_transoform(u, alpha, beta)

    s_bar = tf_stretch(s, interval)
    s_bar_pred = tf_stretch(tf.nn.sigmoid(c), interval)

    # create L0_masted tensor in the graph and tag it with proper names
    l0_tensor_training = tf.identity(tf_hard_sigmoid(s_bar) * tensor,  "training_mask") #
    l0_tensor_prediction = tf.identity(tf_hard_sigmoid(s_bar_pred) * tensor, "prediction_mask")

    add_losses = tf.nn.sigmoid(c - beta * (tf.log(-interval[0]) - tf.log(interval[1])))
    l0_loss = tf.reduce_sum(add_losses)
    add_loss = l0_loss

    return add_loss

def l0_regularizer(scale, scope=None, **kwargs):
    """Returns a function that can be used to apply L2 regularization to weights.
      Small values of L2 can help prevent overfitting the training data.
      Args:
        scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.
        scope: An optional scope name.
      Returns:
        A function with signature `l2(weights)` that applies L2 regularization.
      Raises:
        ValueError: If scale is negative or if scale is not a float.
      """
    if isinstance(scale, numbers.Integral):
        raise ValueError('scale cannot be an integer: %s' % (scale,))
    if isinstance(scale, numbers.Real):
        if scale < 0.:
            raise ValueError('Setting a scale less than 0 on a regularizer: %g.' %
                             scale)
        if scale == 0.:
            logging.info('Scale of 0 disables regularizer.')
            return None # todo: the original output was return lambda _: None; I need to change the format so that I can check when the lambda ==0 without creating new graph

    def l0(weights):
        """Applies l2 regularization to weights."""
        with ops.name_scope(scope, 'l0_regularizer', [weights]) as name:
            my_scale = ops.convert_to_tensor(scale,
                                             dtype=weights.dtype.base_dtype,
                                             name='scale')
            l0_loss = l0_computation(weights, **kwargs)
            return standard_ops.multiply(my_scale, l0_loss, name=name)

    return l0
