import tensorflow as tf
import pprint


def get_weights(shape, name):
    return tf.get_variable(name, shape=shape)


def get_bias(shape, name):
    return tf.zeros(shape=shape, name=name)


def zero_padding(X, pad=(3, 3)):
    paddings = tf.constant([[0, 0], [pad[0], pad[0]],
                            [pad[1], pad[1]], [0, 0]])
    return tf.pad(X, paddings, 'CONSTANT')


def flatten(X):
    return tf.contrib.layers.flatten(X)


def dense(X, out, name):
    in_prev = X.shape.as_list()[1]
    W = get_weights((in_prev, out), name=name+'_W')
    b = get_bias((1, out), name=name+'_b')
    Z = tf.add(tf.matmul(X, W), b, name=name+'_Z')
    A = tf.nn.softmax(Z, name=name)
    params = {'W':W, 'b':b, 'Z':Z, 'A':A}
    return A, params


def conv2D(A_prev, filters, k_size, strides, padding, name):
    m, in_H, in_W, in_C = A_prev.shape.as_list()

    w_shape = (k_size[0], k_size[1], in_C, filters)
    b_shape = (1, 1, 1, filters)

    W = get_weights(shape=w_shape, name=name+'_W')
    b = get_bias(shape=b_shape, name=name+'_b')

    strides = [1, strides[0], strides[1], 1]

    A = tf.nn.conv2d(A_prev, W, strides=strides, padding=padding, name=name)
    params = {'W':W, 'b':b, 'A':A}
    return A, params


def batch_norm(X, name):
    m_, v_ = tf.nn.moments(X, axes=[0, 1, 2], keep_dims=False)
    beta_ = tf.zeros(X.shape.as_list()[3])
    gamma_ = tf.ones(X.shape.as_list()[3])
    bn = tf.nn.batch_normalization(X, mean=m_, variance=v_,
                                   offset=beta_, scale=gamma_,
                                   variance_epsilon=1e-4)
    return bn


def identity_block(X, f, filters, stage, block):
    """
    Implementing a ResNet identity block with shortcut path
    passing over 3 Conv Layers

    @params
    X - input tensor of shape (m, in_H, in_W, in_C)
    f - size of middle layer filter
    filters - tuple of number of filters in 3 layers
    stage - used to name the layers
    block - used to name the layers

    @returns
    A - Output of identity_block
    params - Params used in identity block
    """

    conv_name = 'res' + str(stage) + block + '_branch'
    bn_name = 'bn' + str(stage) + block + '_branch'

    l1_f, l2_f, l3_f = filters

    params = {}

    A1, params[conv_name+'2a'] = conv2D(X, filters=l1_f, k_size=(1, 1), strides=(1, 1),
                                        padding='VALID', name=conv_name+'2a')
    A1_bn = batch_norm(A1, name=bn_name+'2a')
    A1_act = tf.nn.relu(A1_bn)
    params[conv_name+'2a']['bn'] = A1_bn
    params[conv_name+'2a']['act'] = A1_bn

    A2, params[conv_name+'2b'] = conv2D(A1_act, filters=l2_f, k_size=(f, f), strides=(1, 1),
                                        padding='SAME', name=conv_name+'2b')
    A2_bn = batch_norm(A2, name=bn_name+'2b')
    A2_act = tf.nn.relu(A2_bn)
    params[conv_name+'2b']['bn'] = A2_bn
    params[conv_name+'2b']['act'] = A2_act

    A3, params[conv_name+'2c'] = conv2D(A2_act, filters=l3_f, k_size=(1, 1), strides=(1, 1),
                                        padding='VALID', name=conv_name+'2c')
    A3_bn=batch_norm(A3, name=bn_name+'2c')

    A3_add = tf.add(A3_bn, X)
    A = tf.nn.relu(A3_add)
    params[conv_name+'2c']['bn'] = A3_bn
    params[conv_name+'2c']['add'] = A3_add
    params['out'] = A
    return A, params


def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Implementing a ResNet convolutional block with shortcut path
    passing over 3 Conv Layers having different sizes

    @params
    X - input tensor of shape (m, in_H, in_W, in_C)
    f - size of middle layer filter
    filters - tuple of number of filters in 3 layers
    stage - used to name the layers
    block - used to name the layers
    s - strides used in first layer of convolutional block

    @returns
    A - Output of convolutional_block
    params - Params used in convolutional block
    """

    conv_name = 'res' + str(stage) + block + '_branch'
    bn_name = 'bn' + str(stage) + block + '_branch'

    l1_f, l2_f, l3_f = filters

    params = {}

    A1, params[conv_name+'2a'] = conv2D(X, filters=l1_f, k_size=(1, 1), strides=(s, s),
                                        padding='VALID', name=conv_name+'2a')
    A1_bn = batch_norm(A1, name=bn_name+'2a')
    A1_act = tf.nn.relu(A1_bn)
    params[conv_name+'2a']['bn'] = A1_bn
    params[conv_name+'2a']['act'] = A1_bn

    A2, params[conv_name+'2b'] = conv2D(A1_act, filters=l2_f, k_size=(f, f), strides=(1, 1),
                                        padding='SAME', name=conv_name+'2b')
    A2_bn = batch_norm(A2, name=bn_name+'2b')
    A2_act = tf.nn.relu(A2_bn)
    params[conv_name+'2b']['bn'] = A2_bn
    params[conv_name+'2b']['act'] = A2_act

    A3, params[conv_name+'2c'] = conv2D(A2_act, filters=l3_f, k_size=(1, 1), strides=(1, 1),
                                        padding='VALID', name=conv_name+'2c')
    A3_bn=batch_norm(A3, name=bn_name+'2c')
    params[conv_name+'2c']['bn'] = A3_bn

    A_, params[conv_name+'1'] = conv2D(X, filters=l3_f, k_size=(1, 1), strides=(s, s),
                                       padding='VALID', name=conv_name+'1')
    A_bn_ = batch_norm(A_, name=bn_name+'1')

    A3_add = tf.add(A3_bn, A_bn_)
    A = tf.nn.relu(A3_add)
    params[conv_name+'2c']['add'] = A3_add
    params[conv_name+'1']['bn'] = A_bn_
    params['out'] = A
    return A, params


def ResNet50(input_shape=[64, 64, 3], classes=2):

    input_shape=[None]+ input_shape
    params={}

    X_input = tf.placeholder(tf.float32, shape=input_shape, name='input_layer')

    X = zero_padding(X_input, (3, 3))
    params['input'] = X_input
    params['zero_pad'] = X

    # Stage 1
    params['stage1'] = {}
    A_1, params['stage1']['conv'] = conv2D(X, filters=64, k_size=(7, 7), strides=(2, 2),
                                           padding='VALID', name='conv1')
    A_1_bn = batch_norm(A_1, name='bn_conv1')
    A_1_act = tf.nn.relu(A_1_bn)
    A_1_pool = tf.nn.max_pool(A_1_act, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1),
                              padding='VALID')
    params['stage1']['bn'] = A_1_bn
    params['stage1']['act'] = A_1_act
    params['stage1']['pool'] = A_1_pool

    # Stage 2
    params['stage2'] = {}
    A_2_cb, params['stage2']['cb'] = convolutional_block(A_1_pool, f=3, filters=[64, 64, 256],
                                                         stage=2, block='a', s=1)
    A_2_ib1, params['stage2']['ib1'] = identity_block(A_2_cb, f=3, filters=[64, 64, 256],
                                                      stage=2, block='b')
    A_2_ib2, params['stage2']['ib2'] = identity_block(A_2_ib1, f=3, filters=[64, 64, 256],
                                                      stage=2, block='c')

    # Stage 3
    params['stage3'] = {}
    A_3_cb, params['stage3']['cb'] = convolutional_block(A_2_ib2, 3, [128, 128, 512],
                                                         stage=3, block='a', s=2)
    A_3_ib1, params['stage3']['ib1'] = identity_block(A_3_cb, 3, [128, 128, 512],
                                                      stage=3, block='b')
    A_3_ib2, params['stage3']['ib2'] = identity_block(A_3_ib1, 3, [128, 128, 512],
                                                      stage=3, block='c')
    A_3_ib3, params['stage3']['ib3'] = identity_block(A_3_ib2, 3, [128, 128, 512],
                                                      stage=3, block='d')

    # Stage 4
    params['stage4'] = {}
    A_4_cb, params['stage4']['cb'] = convolutional_block(A_3_ib3, 3, [256, 256, 1024],
                                                         stage=4, block='a', s=2)
    A_4_ib1, params['stage4']['ib1'] = identity_block(A_4_cb, 3, [256, 256, 1024],
                                                      stage=4, block='b')
    A_4_ib2, params['stage4']['ib2'] = identity_block(A_4_ib1, 3, [256, 256, 1024],
                                                      stage=4, block='c')
    A_4_ib3, params['stage4']['ib3'] = identity_block(A_4_ib2, 3, [256, 256, 1024],
                                                      stage=4, block='d')
    A_4_ib4, params['stage4']['ib4'] = identity_block(A_4_ib3, 3, [256, 256, 1024],
                                                      stage=4, block='e')
    A_4_ib5, params['stage4']['ib5'] = identity_block(A_4_ib4, 3, [256, 256, 1024],
                                                      stage=4, block='f')

    # Stage 5
    params['stage5'] = {}
    A_5_cb, params['stage5']['cb'] = convolutional_block(A_4_ib5, 3, [512, 512, 2048],
                                                         stage=5, block='a', s=2)
    A_5_ib1, params['stage5']['ib1'] = identity_block(A_5_cb, 3, [512, 512, 2048],
                                                      stage=5, block='b')
    A_5_ib2, params['stage5']['ib2'] = identity_block(A_5_ib1, 3, [512, 512, 2048],
                                                      stage=5, block='c')

    # Average Pooling
    A_avg_pool = tf.nn.avg_pool(A_5_ib2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                                padding='VALID', name='avg_pool')
    params['avg_pool'] = A_avg_pool

    # Output Layer
    A_flat = flatten(A_avg_pool)
    params['flatten'] = A_flat
    A_out, params['out'] = dense(A_flat, classes, name='fc'+str(classes))

    return A_out, params


if __name__ == '__main__':
    A, params = ResNet50()
    pprint.pprint(params, stream=open('ResNet50.json', 'w'), indent=2)
    #pp.pprint(params)

