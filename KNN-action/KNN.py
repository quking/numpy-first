from numpy import *


# 首先处理文件为二维数组方便处理


class KNN():

    def fileToMatrix(self, filename):  # 这个函数只针对datingTestSet数据集处理
        f = open(filename)
        lens = len(f.readlines())
        dataset = zeros((lens, 3))
        label = zeros((lens, 1))
        index = 0
        f = open(filename)  # 之前已经被打开过一次，需要重新打开
        for fileline in f.readlines():
            fileline = fileline.strip()
            fileresult = fileline.split("\t")
            dataset[index, :] = fileresult[0:3]
            label[index] = fileresult[-1]
            index += 1
        return dataset, label

    def auto_normal(self, dataset):
        """

        :param dataset:
        :return:
        """


if __name__ == "__main__":
    k = KNN()
    dating, lab = k.fileToMatrix("datingTestSet2.txt")
    print(lab)
