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
        这里用到了公式进行常规化，因为不同的数据差别太大了，进行运算不方便
        (x-min)/max-min
        :param dataset:
        :return:
        """
        mincolumn = dataset.min(0)
        maxcolumn = dataset.max(0)
        m = dataset.shape[0]
        mindataset = tile(m, mincolumn)
        maxdataset = tile(m, maxcolumn)
        newdataset = (dataset - mindataset) / tile(m, maxdataset - mindataset)
        return newdataset

    def classfy(self, inX, dataset, label, k):
        """
        这里对inX进行KNN分类
        :param inX: 待分类数据，这里只有一行数据，需要进行tile
        :param dataset: 训练数据集,这里的数据集假设是已经normal好的，后面针对具体问题的时候，会先进行normal，再来调用这个函数
        :param label: 训练标签
        :param k: K个点
        :return: 返回分类类型
        """
        m = dataset.shape[0]
        diffdata = dataset - tile(m, inX)
        sumdata = (diffdata ** 2).sum(1)  # 这里0是对列求和，1是对行求和
        sumdata = sumdata ** 0.5
        indexs = argsort(sumdata)  # 这里返回排序好的下标
        labelset = []
        for i in range(k):
            labelset.append(label[indexs[i]])
        return max(labelset, key=labelset.count)  # 返回labelset中最多的元素


if __name__ == "__main__":
    """
    下面是对实例进行计算
    """
    k = KNN()
    dating, lab = k.fileToMatrix("datingTestSet2.txt")
    dating = k.auto_normal(dating)
