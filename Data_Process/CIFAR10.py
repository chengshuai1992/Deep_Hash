# https://blog.csdn.net/qq_41635631/article/details/79784391
# https://blog.csdn.net/zeuseign/article/details/72773342
import pickle
import numpy as np
import os


class Readecifar10():

    def __init__(self, dir_path, onehot=True):
        self.dir_path = dir_path
        self.onehot = onehot

    def load_batch_cifar10(self, cifar_file):
        cifar_file = open(cifar_file, 'rb')
        batch = pickle.load(cifar_file, encoding='latin1')
        cifar_file.close()
        image = batch['data']
        label = batch['labels']
        return image, label

    def load_cifar10(self):
        image = []
        label = []
        for i in range(1):
            file_path = "data_batch_" + str(i + 1)
            cifar_file = os.path.join(self.dir_path, file_path)
            data_image, data_label = self.load_batch_cifar10(cifar_file)

            if (len(data_image) == len(data_label)):
                for i in range(len(data_image)):
                    image.append(data_image[i])
                    label_hot=np.zeros(10)
                    label_hot[int(data_label[i])]=1
                    label.append(label_hot)

        return image, label


if __name__ == '__main__':
    image, label = Readecifar10('/home/cheng/Data/cifar-10-batches-py').load_cifar10()
    print(label)
