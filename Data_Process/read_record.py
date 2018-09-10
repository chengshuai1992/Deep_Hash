import read_cifar10_data
import numpy as np
import tensorflow as tf
import os
import cv2

####数据的预处理部分已经差不多完成了
"""
输入文件队列：这里是处理文件输入并且最后把数据整理成一个batch
tf.train.match_filenames_once试图读取多个文件 使用正则表达式获得文件名
tf.trian.string_input_producer 对文件列表创建输入队列
在这里要做的工作是生成Tfrecord文件，然后可以实现把文件组成batch
"""

config = {

    'batch_size': 200,
    'min_after_dequeue': 1000,

}

match_files="/home/cheng/Data/train-*"

def reader_TFrecord():

    files = tf.train.match_filenames_once(match_files)

    filename_queue = tf.train.string_input_producer(files, shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    feature = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        })
    image_raw = tf.decode_raw(feature['image'], tf.uint8)
    image = tf.reshape(image_raw, [227, 227, 3])
    label = tf.cast(feature['label'], tf.int32)
    label = tf.one_hot(label, 10, 1, 0)

    return image, label


def next_batch(image, label):
    min_after_dequeue = config['min_after_dequeue']
    batch_size = config['batch_size']
    capacity = min_after_dequeue + 3 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size,
                                                      capacity=capacity, min_after_dequeue=min_after_dequeue)

    return image_batch, label_batch


if __name__ == '__main__':


    image, label = reader_TFrecord()

    init = (tf.global_variables_initializer(), tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    image_batch, label_batch = next_batch(image, label)
    with tf.Session() as sess:

        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(2):
            image, label = sess.run([image_batch, label_batch])

            for i in range(10):
                print(label[i])
                cv2.imshow("picture", image[i])
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        coord.request_stop()
        coord.join(threads)