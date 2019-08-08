# L_0 Regularizer, l0_layer and l0_dense:
## Introduction: 
- L_0 regularization has only zero value at the origin and holds fixed otherwise. The regularization form can discard unnecessary tensor by forcing them to zero while keeping others untouched at the same time. However it is not directly applicable since there is no gradient almost everywhere and the most informative point is not differentiable.
- We implement an approximate version of L_0 regularization from [Louizos et al. 2017](https://arxiv.org/abs/1712.01312).
- We provide examples (l0_dense and l0_layer) to demonstrate how to incoporate L_0 regularization into a model since L_0 regularization induces an architectural change (probabilistic mask creation) therefore cannot be applied directly as any other L^p regularization 

## Detailed Introduction:
- the program is organized as the following: 
	- **l0_computation -> l0_regularizer -> l0_dense/l0_layer**
	- **l0_computation** defines the computational mechanisms of l0 regularization on a tensors during which a masked conditional tensor is created and will replace the original tensor for model building. For details please refer to [Louizos et al. 2017](https://arxiv.org/abs/1712.01312). **l0_computation** is not recommended for users for common applications 
- **l0_regularizer**:
	- adpated from the structure of tf.contrib.layers.l2_regularizer
	- it is intended to develop into a format that can be applicable to any architecture
	- inputs: scale, scope=None
	- outputs: the (scaled) regularization loss
- **l0_dense**
	- adapted from the structure of tf.layers.dense
	- almost the same as the original dense function.
	- Except an additional argument is_training is added for tagging training/prediction status of the weights/bias
	- the losses created after applying l0_regularizer can be obtained by calling _tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)_
- **l0_layer**
	- inherited from the base.Layer class and structured much like tf.layers.dropout
	- input format is assumed to be [?, D] or [D, ?]
	- inputs: input_tensor, reg_const, training, seed, name
	- outputs: the masked layer activity
	- the losses created after applying l0_regularizer can be obtained by calling _tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)_

## Requirements:
- python 3.6
- tensorflow 1.7

## Installation and Usage:
- python3 -m pip install --index-url https://test.pypi.org/simple/ l0-regularization
- In the code (also for sanity check):
    - from l0_regularization.l0_layer import l0_layer 
    - from l0_regularization.l0_dense import l0_dense
- Checkout the demo_with_autoencoder.ipynb for an application of our module to autoencoders

## Note:
- The masks are created in l0_computation and are retrieved at the l0_dense.call method.
- The PyPI page for the project https://test.pypi.org/project/l0-regularization/

## To Solve/Do:
- Check. The training loss is too large after the insertion of L_0, not sure if it is normal (at Swiss-Roll Dataset)
- Do(optional). Make possible for using more than one regularization mechanisms
- Why? failure to identify the variable created in "collection" argument in tf...dense to be initialized
- How? Better way to retrive the Tensors from l0_dense.call(...)
- How? Minimize the argument for l0_dense (regularizer_kernel, _bias can be reduced to just reg_const)
- Make it better:
	- error handling, may resort to try/except and RaiseError
	- Output None istead of lambda _: None for l0_regularizer, for later check and avoidance of unecessary graph creation. is it possible to do this with keeping the later format?
	- More efficient creation for L0MLP (not in this project) using better OOP ideas.
- Discussion. 1. broadcasting for l0_layer; 2. the way to check the tensor_shape in l0_computation; 3. l0_dense may not make sense to have a seed since the variable initializer have no seed neither.

