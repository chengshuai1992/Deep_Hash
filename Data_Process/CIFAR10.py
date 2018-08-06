#https://blog.csdn.net/qq_41635631/article/details/79784391
import pickle
import numpy as np
import os


class Readecifar10():
    def __init__(self,dir_path,onehot=True):
        self.dir_path=dir_path
        self.onehot=onehot


    def load_batch_cifar10(self,cifar_file):

        cifar_file=open(cifar_file,'rb')
        batch=pickle.load(cifar_file,encoding='latin1')
        cifar_file.close()
        image=batch['data']
        label=batch['labels']
        return image,label
    def load_cifar10(self):
        image=[]
        label=[]
        for i in range(1):
            file_path="data_batch_"+str(i+1)
            cifar_file=os.path.join(self.dir_path,file_path)

            data_image,data_label=self.load_batch_cifar10(cifar_file)

            return data_image,data_label


if __name__ == '__main__':
        data1,data2=Readecifar10('/home/cheng/Data/cifar-10-batches-py').load_cifar10()
        print(data1,data2)