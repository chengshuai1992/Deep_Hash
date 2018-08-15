# https://blog.csdn.net/zhq0808/article/details/78482266

import CIFAR10
import numpy as np
import tensorflow as tf
import os
import cv2

"""
输入文件队列：这里是处理文件输入并且最后把数据整理成一个batch
tf.train.match_filenames_once试图读取多个文件 使用正则表达式获得文件名
tf.trian.string_input_producer 对文件列表创建输入队列
在这里要做的工作是生成Tfrecord文件，然后可以实现把文件组成batch
"""

config = {
    'original_file_dir': "/home/cheng/Data/cifar-10-batches-py/",
    'output_data_dir': "/home/cheng/Data/cifar-train_data/",
    'num_shards': 1,
    'instances_per_shard': 10000,
    'match_files': "/home/cheng/Data/cifar-train_data/train-*",
    'width': 32,
    'height': 32,

}


def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def data_reshape():
    ### 这个函数的作用是对得到的原始图像数据进行多种变换。在这里现不做这部分工作。
    ### original_file_path=/home/cheng/Data/cifar-10-batches-py
    # 读取所有的数据
    images, labels = CIFAR10.Readecifar10(config["original_file_dir"]).load_cifar10()

    return images, labels


def writer_TFrecord():
    images, labels = data_reshape()

    if (len(images) == len(labels)):
        for i in range(int(config['num_shards'])):
            output_filename = "%s-%.5d-of-%.5d" % ("train", i, config['num_shards'])
            out_file = os.path.join(config['output_data_dir'], output_filename)
            writer = tf.python_io.TFRecordWriter(out_file)
            for j in range(config['instances_per_shard']):
                # shape = np.prod(images[i * config['instances_per_shard'] + j].shape[0:])
                # image = np.array(images[i * config['instances_per_shard'] + j]).reshape((1, shape))
                image_raw = images[i * config['instances_per_shard'] + j].tobytes()
                label = labels[i * config['instances_per_shard'] + j]
                data = tf.train.Example(features=tf.train.Features(feature={
                    "image": _bytes_feature(image_raw),
                    "label": _int64_feature(label)
                }))

                writer.write(data.SerializeToString())

            writer.close()


def reader_TFrecord():
    width = config['width']
    height = config['height']
    files = tf.train.match_filenames_once(config['match_files'])
    filename_queue = tf.train.string_input_producer(files, shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    feature = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        })
    image_raw = tf.decode_raw(feature['image'], tf.uint8)
    image = tf.reshape(image_raw, [width, height, 3])
    label = tf.cast(feature['label'], tf.int32)


    return image, label


if __name__ == '__main__':
    #writer_TFrecord()
    image, label = reader_TFrecord()
    init = (tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess)
        image, label = sess.run([image, label])
        print(image)
        label_hot = np.zeros(10)
        label_hot[int(label)] = 1
        print(label_hot)
        cv2.imshow('picture', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
