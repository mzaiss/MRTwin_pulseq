from tensorflow.python.ops import nn
from tensorflow.python.layers import base
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import utils
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import ops
from l0_regularization.l0_utils import l0_regularizer
from l0_regularization.utils import get_l0_maskeds



class L0_Layer(base.Layer):
  """Applies Dropout to the input.
  Dropout consists in randomly setting a fraction `rate` of input units to 0
  at each update during training time, which helps prevent overfitting.
  The units that are kept are scaled by `1 / (1 - rate)`, so that their
  sum is unchanged at training time and inference time.
  Arguments:
    rate: The dropout rate, between 0 and 1. E.g. `rate=0.1` would drop out
      10% of input units.
    noise_shape: 1D tensor of type `int32` representing the shape of the
      binary dropout mask that will be multiplied with the input.
      For instance, if your inputs have shape
      `(batch_size, timesteps, features)`, and you want the dropout mask
      to be the same for all timesteps, you can use
      `noise_shape=[batch_size, 1, features]`.
    seed: A Python integer. Used to create random seeds. See
      @{tf.set_random_seed}.
      for behavior.
    name: The name of the layer (string).
  """

  def __init__(self, reg_const=1e-5,
               seed=None, # todo-1, the seed function is not yet added
               name=None,
               **kwargs):
    super().__init__(name=name, **kwargs)
    self.reg_const = reg_const
    self.seed = seed

  def call(self, inputs, training=False):
      l0_op = l0_regularizer(self.reg_const, seed=self.seed)

      if l0_op is not None:
          self.add_loss(l0_op(inputs))
          l0_regularizer_scope = self.scope_name + '/l0_regularizer/'
          layer_trng, layer_pred = get_l0_maskeds(l0_regularizer_scope)
          return utils.smart_cond(training,
                                  lambda: layer_trng,
                                  lambda: layer_pred)
      else:
          return inputs

  # there was compute_output_shape from tf.layers.dropout which I deleted, note here for future problem



def l0_layer(inputs,
             reg_const=1e-5,
             training=False,
             seed=None,
             name=None):
  """Applies Dropout to the input.
  Dropout consists in randomly setting a fraction `rate` of input units to 0
  at each update during training time, which helps prevent overfitting.
  The units that are kept are scaled by `1 / (1 - rate)`, so that their
  sum is unchanged at training time and inference time.
  Arguments:
    inputs: Tensor input.
    rate: The dropout rate, between 0 and 1. E.g. "rate=0.1" would drop out
      10% of input units.
    noise_shape: 1D tensor of type `int32` representing the shape of the
      binary dropout mask that will be multiplied with the input.
      For instance, if your inputs have shape
      `(batch_size, timesteps, features)`, and you want the dropout mask
      to be the same for all timesteps, you can use
      `noise_shape=[batch_size, 1, features]`.
    seed: A Python integer. Used to create random seeds. See
      @{tf.set_random_seed}
      for behavior.
    training: Either a Python boolean, or a TensorFlow boolean scalar tensor
      (e.g. a placeholder). Whether to return the output in training mode
      (apply dropout) or in inference mode (return the input untouched).
    name: The name of the layer (string).
  Returns:
    Output tensor.
  Raises:
    ValueError: if eager execution is enabled.
  """
  layer = L0_Layer(reg_const, seed=seed, name=name)
  return layer.apply(inputs, training=training)
