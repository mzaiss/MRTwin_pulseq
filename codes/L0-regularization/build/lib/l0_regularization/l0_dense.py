from __future__ import division, print_function, absolute_import
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops import init_ops
from l0_regularization.l0_utils import l0_regularizer
from l0_regularization.utils import get_l0_maskeds




class L0Dense(Dense):
  def __init__(self, is_training, seed, *args, **kwargs):
    self.is_training = is_training
    self.seed = seed
    super().__init__(*args, **kwargs)


  def call(self, inputs):
      l0_relative_path = '/Regularizer/l0_regularizer/'
      l0_absolute_path = self.scope_name

      # create masked kernel/bias
      kernel_0 = self.kernel
      if self.kernel_regularizer is not None:
          l0_masked_kernels_path = l0_absolute_path + "/kernel" + l0_relative_path
          trng_masked_kernel, pred_masked_kernel = get_l0_maskeds(l0_masked_kernels_path)
          masked_kernel = tf.cond(self.is_training,
                                  lambda: trng_masked_kernel,
                                  lambda: pred_masked_kernel, name='l0_masked_kernel')

          self.kernel = masked_kernel


      bias_0 = self.bias
      if self.bias_regularizer is not None:
          l0_masked_bias_path = l0_absolute_path + "/bias" + l0_relative_path
          trng_masked_bias, pred_masked_bias = get_l0_maskeds(l0_masked_bias_path)
          masked_bias = tf.cond(self.is_training,
                                lambda: trng_masked_bias,
                                lambda: pred_masked_bias, name='l0_masked_bias')

          self.bias = masked_bias

      output = super().call(inputs)

      self.kernel = kernel_0
      self.bias = bias_0

      return output




def l0_dense(
      inputs, units, is_training=False, seed=None,
      activation=None,
      use_bias=True,
      kernel_initializer=None,
      bias_initializer=init_ops.zeros_initializer(),
      kernel_regularization_scale=1e-6,
      bias_regularization_scale=1e-6,
      activity_regularizer=None,
      kernel_constraint=None,
      bias_constraint=None,
      trainable=True,
      name=None,
      reuse=None):

    """Functional interface for the l0 densely-connected layer (adapted from tf.layers.dense).

      This layer implements the operation:
      `outputs = activation(inputs.kernel + bias)`
      Where `activation` is the activation function passed as the `activation`
      argument (if not `None`), `kernel` is a weights matrix created by the layer,
      and `bias` is a bias vector created by the layer
      (only if `use_bias` is `True`).

      Arguments:
        inputs: Tensor input.
        units: Integer or Long, dimensionality of the output space.
        is_training: Bool, Tag the module for checking if it is at training or prediction stage, and then deploy different weights
        seed: A Python integer. Used to create random seeds. See
          @{tf.set_random_seed}
          for behavior.
        activation: Activation function (callable). Set it to None to maintain a
          linear activation.
        use_bias: Boolean, whether the layer uses a bias.
        kernel_initializer: Initializer function for the weight matrix.
          If `None` (default), weights are initialized using the default
          initializer used by `tf.get_variable`.
        bias_initializer: Initializer function for the bias.
        kernel_regularization_scale: Long, l0 regularization scale for the weight matrix.
        bias_regularization_scale: Long, l0 regularization scale for the bias .
        activity_regularizer: Regularizer function for the output.
        kernel_constraint: An optional projection function to be applied to the
            kernel after being updated by an `Optimizer` (e.g. used to implement
            norm constraints or value constraints for layer weights). The function
            must take as input the unprojected variable and must return the
            projected variable (which must have the same shape). Constraints are
            not safe to use when doing asynchronous distributed training.
        bias_constraint: An optional projection function to be applied to the
            bias after being updated by an `Optimizer`.
        trainable: Boolean, if `True` also add variables to the graph collection
          `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
        name: String, the name of the layer.
        reuse: Boolean, whether to reuse the weights of a previous layer
          by the same name.

      Returns:
        Output tensor the same shape as `inputs` except the last dimension is of
        size `units`.

      Raises:
        ValueError: if eager execution is enabled.
      """
    kernel_regularizer = l0_regularizer(kernel_regularization_scale)
    bias_regularizer = l0_regularizer(bias_regularization_scale)
    layer = L0Dense(is_training,
                    seed,
                    units,
                    activation=activation,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,
                    kernel_constraint=kernel_constraint,
                    bias_constraint=bias_constraint,
                    trainable=trainable,
                    name=name,
                    dtype=inputs.dtype.base_dtype,
                    _scope=name,
                    _reuse=reuse)
    return layer.apply(inputs)




if __name__ == "__main__":

    # the 1st test
    n_input = 784
    n_hidden_1 = 256
    c_l0 = 1e-3
    x = tf.placeholder("float", [None, n_input])
    is_training = True
    tmp = l0_dense(x, n_hidden_1, is_training=is_training, kernel_regularization_scale=0.0)

