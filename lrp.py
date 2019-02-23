from tensorflow.python.ops import nn_ops, gen_nn_ops
import tensorflow as tf
import numpy as np
import gc

def backprop_conv(alpha, activation, kernel, bias, relevance, strides, padding='VALID'):
    W_p = tf.maximum(0., kernel)
    b_p = tf.maximum(0., bias)
    z_p = nn_ops.conv2d(activation, W_p, strides, padding) + b_p
    s_p = relevance / z_p
    c_p = nn_ops.conv2d_backprop_input(tf.shape(activation), W_p, s_p, strides, padding)

    W_n = tf.minimum(0., kernel)
    b_n = tf.minimum(0., bias)
    z_n = nn_ops.conv2d(activation, W_n, strides, padding) + b_n
    s_n = relevance / z_n
    c_n = nn_ops.conv2d_backprop_input(tf.shape(activation), W_n, s_n, strides, padding)

    return activation * (alpha * c_p + (1 - alpha) * c_n)

def backprop_pool(activation, relevance, ksize, strides, pooling_type, padding='VALID'):
    if pooling_type.lower() is 'avg': # avg pooling
        z = nn_ops.avg_pool(activation, ksize, strides, padding) + 1e-10
        s = relevance / z
        c = gen_nn_ops._avg_pool_grad(tf.shape(activation), s, ksize, strides, padding)
        return activation * c
    else: # max pooling
        z = nn_ops.max_pool(activation, ksize, strides, padding) + 1e-10
        s = relevance / z
        c = gen_nn_ops.max_pool_grad(activation, z, s, ksize, strides, padding)
        return activation * c

def backprop_dense(alpha, activation, kernel, bias, relevance):
    W_p = tf.maximum(0., kernel)
    b_p = tf.maximum(0., bias)
    z_p = tf.matmul(activation, W_p) + b_p
    s_p = relevance / z_p
    c_p = tf.matmul(s_p, tf.transpose(W_p))

    W_n = tf.minimum(0., kernel)
    b_n = tf.minimum(0., bias)
    z_n = tf.matmul(activation, W_n) + b_n
    s_n = relevance / z_n
    c_n = tf.matmul(s_n, tf.transpose(W_n))

    return activation * (alpha * c_p + (1 - alpha) * c_n)


def lrp(backprop_layers, data, sess, nnModel):
    """ pool_biases is required only if layer_count is more than one """
    #backprop layers acquired with lrp_layers method
    explained = []
    expl_ = sess.run(backprop_layers[-1], feed_dict = {nnModel.nn_vector_inputs:data, nnModel.isTraining:False})
    explained.append(expl_.reshape((128,300)))

    return explained

def lrp_layers(alpha, layer_count, activations, weights, biases, pool_biases=None):
    #https://github.com/1202kbs/Understanding-NN
    range_ = 1 # DO NOT FORGET TO CHANGE THIS TO 10 # update: no need to change, 1 is ok
    for y in range(layer_count):
        layers = []
        if layer_count == 1:
            pool_bias = [1,126,1,1]
        elif layer_count == 3:
            activations = activations_splitted[y]
            biases = biases_splitted[y]
            weights = weights_splitted[y]
            pool_bias = pool_biases[y]
        for x in range(range_):
            Rs = []
            j = 0
            logit = x
            for i in range(len(activations) - 1):
    #             print("i: {}, activations[i]: {} ".format(i, activations[i]))
                if i is 0:
                    Rs.append(activations[i][:,logit,None])
                    Rs.append(backprop_dense(alpha, activations[i + 1], weights[j][:,logit,None], biases[j][logit,None], Rs[-1]))
                    j += 1
                    continue

                elif 'fc' in activations[i].name.lower():
    #                 print("dense")
                    Rs.append(backprop_dense(alpha, activations[i + 1], weights[j], biases[j], Rs[-1]))
                    j += 1
                elif 'reshape' in activations[i].name.lower():
    #                 print("reshape")
                    shape = activations[i + 1].get_shape().as_list()
                    shape[0] = -1
                    Rs.append(tf.reshape(Rs[-1], shape))
                elif 'preblock' in activations[i].name.lower():
    #                 print("preblock (conv)")
                    Rs.append(backprop_conv(alpha, activations[i + 1], weights[j], biases[j], Rs[-1], (1,1,1,1)))
                    j += 1
                elif 'pool' in activations[i].name.lower():
    #                 print("pool")
                    if 'max' in activations[i].name.lower():
                        pooling_type = 'max'
                    else:
                        pooling_type = 'avg'
                    Rs.append(backprop_pool(activations[i + 1], Rs[-1], pool_bias, [1,1,1,1], pooling_type="max"))
                else:
                    pass

            layers.append(Rs[-1])

    return layers

def get_word_relevances(alpha, backprop_layers, layer_count, data, dataRaw, sess, nnModel, activations, weights, biases):
    """example: word_importances = lrp.get_word_relevances(alpha, layer_count, batch_x[0:1], trainData[0], sess, nnModel, activations, weights, biases)"""

    explained = lrp(backprop_layers, data, sess, nnModel)

    results_combined = []
    for i in range(layer_count):
        explained[i] = [np.mean(a) for a in explained[i]]

    if layer_count == 1:
        results_combined = explained[0]
    elif layer_count == 3:
        results_combined = np.add(explained[0], explained[1])
        results_combined = np.add(results_combined, explained[2])

    # Conver nans to 0
    for i in range(len(results_combined)):
        if str(results_combined[i]) == "nan":
               results_combined[i] = 0
                
    # Normalize
    norm = np.linalg.norm(results_combined)
    try:
        results_combined = results_combined / norm
    except:
        pass

    # Match words with their relevance
    word_importances = list(zip(dataRaw, results_combined))

    del explained
    del norm
    gc.collect()

    return word_importances, results_combined