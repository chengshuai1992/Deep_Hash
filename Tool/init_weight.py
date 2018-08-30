import numpy as np
import tensorflow as tf

net_data = {}
deep_param_img = {}

def save_npy():
    ### Conv1
    ### Output 96, kernel 11
    kernel_cov1 = tf.Variable(tf.truncated_normal([11, 11, 3, 96], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    biases_cov1 = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32),
                         trainable=True, name='biases')
    ### Conv2
    ### Output 256,  kernel 5
    kernel_cov2 = tf.Variable(tf.truncated_normal([5, 5, 96, 256], dtype=tf.float32,
                                                  stddev=1e-1), name='weights')
    biases_cov2 = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                              trainable=True, name='biases')



    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        kernel = sess.run(kernel)
        biases = sess.run(biases)
    net_data['conv1'] = [kernel, biases]
    np.save("model_weights", net_data)

    print(net_data['conv1'][0])
    print(net_data['conv1'][1])
    print(net_data.keys())


def load_npy():
    net_data = np.load("model_weights.npy").item()
    kernel = tf.Variable(net_data['conv1'][0], name='weights')
    biases = tf.Variable(net_data['conv1'][1], name='biases')
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        kernel = sess.run(kernel)
        biases = sess.run(biases)

    deep_param_img['conv1'] = [kernel, biases]

    print(kernel)
    print(biases)


if __name__ == '__main__':
    save_npy()
    load_npy()
