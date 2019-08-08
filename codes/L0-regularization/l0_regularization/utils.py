import tensorflow as tf


def get_tensor_by_name(name):
    return tf.get_default_graph().get_tensor_by_name(name)

def get_l0_maskeds(scope_name):
    trng_masked_tensor = get_tensor_by_name(scope_name + "training_mask:0")
    pred_masked_tensor = get_tensor_by_name(scope_name + "prediction_mask:0")

    return trng_masked_tensor, pred_masked_tensor

