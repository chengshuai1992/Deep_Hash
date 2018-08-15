import tensorflow as tf
import numpy as np


def alexnet_layer():
    deep_param_img = {}
    train_layers = []
    train_last_layer = []
    print("loading img model")
    net_data = np.load(model_weights).item()
    """
    model_weights要初始化
    distorted_image 是要作变换的数据
    input_data 227*227
    output_dim 输出维度 分类是10
    根据论文从第二个卷积开始数据分成两组，在这里使用一个GPU所以修改模型，只有一组。
    """


    ###这里还有一个数据处理步骤



    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(net_data['conv1'][0], name='weights')
        conv = tf.nn.conv2d(distorted_image, kernel, [1, 4, 4, 1], padding='VALID')
        biases = tf.Variable(net_data['conv1'][1], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(out, name=scope)
        deep_param_img['conv1'] = [kernel, biases]
        train_layers += [kernel, biases]

        ### Pool1
    pool1 = tf.nn.max_pool(conv1,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool1')

    ### LRN1
    radius = 2;
    alpha = 2e-05;
    beta = 0.75;
    bias = 1.0
    lrn1 = tf.nn.local_response_normalization(pool1,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    ### Conv2
    ### Output 256, pad 2, kernel 5, group 2
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(net_data['conv2'][0], name='weights')
        conv = tf.nn.conv2d(lrn1, kernel, [1, 1, 1, 1], padding='SAME')

        """
        group = 2
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
        input_groups = tf.split(3, group, lrn1)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        ### Concatenate the groups
        conv = tf.concat(3, output_groups)
        """
        biases = tf.Variable(net_data['conv2'][1], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(out, name=scope)
        deep_param_img['conv2'] = [kernel, biases]
        train_layers += [kernel, biases]

    ### Pool2
    pool2 = tf.nn.max_pool(conv2,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool2')

    ### LRN2
    radius = 2;
    alpha = 2e-05;
    beta = 0.75;
    bias = 1.0
    lrn2 = tf.nn.local_response_normalization(pool2,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    ### Conv3
    ### Output 384, pad 1, kernel 3
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(net_data['conv3'][0], name='weights')
        conv = tf.nn.conv2d(lrn2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv3'][1], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(out, name=scope)
        deep_param_img['conv3'] = [kernel, biases]
        train_layers += [kernel, biases]

    ### Conv4
    ### Output 384, pad 1, kernel 3, group 2
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(net_data['conv4'][0], name='weights')
        """
        group = 2
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
        input_groups = tf.split(3, group, conv3)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        ### Concatenate the groups
        conv = tf.concat(3, output_groups)
        """

        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv4'][1], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(out, name=scope)
        deep_param_img['conv4'] = [kernel, biases]
        train_layers += [kernel, biases]

    ### Conv5
    ### Output 256, pad 1, kernel 3, group 2
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(net_data['conv5'][0], name='weights')
        """
        group = 2
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
        input_groups = tf.split(3, group, conv4)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        ### Concatenate the groups
        conv = tf.concat(3, output_groups)
        """
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv5'][1], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(out, name=scope)
        deep_param_img['conv5'] = [kernel, biases]
        train_layers += [kernel, biases]

    ### Pool5
    pool5 = tf.nn.max_pool(conv5,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool5')

    ### FC6
    ### Output 4096
    with tf.name_scope('fc6') as scope:
        shape = int(np.prod(pool5.get_shape()[1:]))
        fc6w = tf.Variable(net_data['fc6'][0], name='weights')
        fc6b = tf.Variable(net_data['fc6'][1], name='biases')
        pool5_flat = tf.reshape(pool5, [-1, shape])
        fc5 = pool5_flat
        fc6l = tf.nn.bias_add(tf.matmul(pool5_flat, fc6w), fc6b)
        fc6_drop = tf.nn.dropout(tf.nn.relu(fc6l), 0.5)
        fc6 = tf.nn.relu(fc6l)
        deep_param_img['fc6'] = [fc6w, fc6b]
        train_layers += [fc6w, fc6b]

    ### FC7
    ### Output 4096
    with tf.name_scope('fc7') as scope:
        fc7w = tf.Variable(net_data['fc7'][0], name='weights')
        fc7b = tf.Variable(net_data['fc7'][1], name='biases')
        fc7l = tf.nn.bias_add(tf.matmul(fc6_drop, fc7w), fc7b)
        fc7_drop = tf.nn.dropout(tf.nn.relu(fc7l), 0.5)
        fc7lo = tf.nn.bias_add(tf.matmul(fc6, fc7w), fc7b)
        fc7 = tf.nn.relu(fc7lo)
        deep_param_img['fc7'] = [fc7w, fc7b]
        train_layers += [fc7w, fc7b]

    ### FC8
    ### Output output_dim
    with tf.name_scope('fc8') as scope:
        ### Differ train and val stage by 'fc8' as key
        if 'fc8' in net_data:
            fc8w = tf.Variable(net_data['fc8'][0], name='weights')
            fc8b = tf.Variable(net_data['fc8'][1], name='biases')
        else:
            fc8w = tf.Variable(tf.random_normal([4096, output_dim],
                                                dtype=tf.float32,
                                                stddev=1e-2), name='weights')
            fc8b = tf.Variable(tf.constant(0.0, shape=[output_dim],
                                           dtype=tf.float32), name='biases')
        fc8l = tf.nn.bias_add(tf.matmul(fc7_drop, fc8w), fc8b)
        fc8_drop = tf.nn.tanh(fc8l)
        fc8lo = tf.nn.bias_add(tf.matmul(fc7, fc8w), fc8b)
        fc8 = tf.nn.tanh(fc8lo)
        deep_param_img['fc8'] = [fc8w, fc8b]
        train_last_layer += [fc8w, fc8b]

    return fc8_drop, fc8
