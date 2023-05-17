import numpy as np
import scipy.io as sio
from sklearn import preprocessing


class HSI_dataset(object):
    """docstring for HSI_dataset"""

    def __init__(self, Scale=100):
        super(HSI_dataset, self).__init__()
        self.Scale = Scale
        self.train_ratio = 0.01
        self.val_ratio = 0.01
        self.sample = 'universe'
        self.sample_ratio = 0.5
        self.max_epoch = 100
        self.learning_rate = 0
        self.class_count = 0

    def data_select(self, FLAG):
        if FLAG == 1:
            data_mat = sio.loadmat('./data/Indian_pines_corrected.mat')
            data = data_mat['indian_pines_corrected']
            gt_mat = sio.loadmat('./data/Indian_pines_gt.mat')
            gt = gt_mat['indian_pines_gt']
            # 参数预设
            self.train_ratio = 0.05
            self.val_ratio = 0.01
            class_count = 16  # 样本类别数
            # rgb = [43, 21, 11]
            # learning_rate = 8e-4  # 学习率
            self.learning_rate = 8e-4
            # max_epoch =600  # 迭代次数
            # pca_components = 32
            # fraction = 0.98
            dataset_name = "indian_"  # 数据集名称

            pass
        if FLAG == 2:
            data_mat = sio.loadmat('./data/PaviaU.mat')
            data = data_mat['paviaU']
            gt_mat = sio.loadmat('./data/PaviaU_gt.mat')
            gt = gt_mat['paviaU_gt']

            # 参数预设
            self.train_ratio = 0.01
            self.val_ratio = 0.01
            class_count = 9  # 样本类别数
            # rgb = [55, 41, 12]
            # learning_rate = 9e-4  # 学习率
            self.learning_rate = 9e-4
            # max_epoch = 600  # 迭代次数
            # fraction = 0.999
            # pca_components = 18
            dataset_name = "paviaU_"  # 数据集名称

            pass
        if FLAG == 3:
            data_mat = sio.loadmat('./data/Salinas_corrected.mat')
            data = data_mat['salinas_corrected']
            gt_mat = sio.loadmat('./data/Salinas_gt.mat')
            gt = gt_mat['salinas_gt']

            # 参数预设
            self.train_ratio = 0.01
            self.val_ratio = 0.01
            class_count = 16  # 样本类别数
            rgb = [29, 19, 9]
            # learning_rate = 9e-4  # 学习率
            self.learning_rate = 5e-4
            # max_epoch = 600  # 迭代次数
            # pca_components = 32
            dataset_name = "salinas_"  # 数据集名称

            pass
        if FLAG == 4:
            data_mat = sio.loadmat('./data/KSC.mat')
            data = data_mat['KSC']
            gt_mat = sio.loadmat('./data/KSC_gt.mat')
            gt = gt_mat['KSC_gt']

            # 参数预设
            self.train_ratio = 0.05
            self.val_ratio = 0.01
            class_count = 13  # 样本类别数
            rgb = [43, 21, 11]
            # learning_rate = 5e-4  # 学习率
            # max_epoch = 600  # 迭代次数
            pca_components = 26
            dataset_name = "KSC_"  # 数据集名称

            pass
        if FLAG == 5:
            data_mat = sio.loadmat('./data/Botswana.mat')
            data = data_mat['Botswana']
            gt_mat = sio.loadmat('./data/Botswana_gt.mat')
            gt = gt_mat['Botswana_gt']

            # 参数预设
            self.train_ratio = 0.05
            self.val_ratio = 0.01
            class_count = 14  # 样本类别数
            rgb = [75, 33, 15]
            # learning_rate = 5e-4  # 学习率
            # max_epoch = 600  # 迭代次数
            pca_components = 28
            dataset_name = "Botswana_"  # 数据集名称

            pass
        return data, gt, class_count, dataset_name
    def GT_one_hot(self, gt, class_count):
        GT_one_hot = []
        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                temp = np.zeros(class_count, dtype=np.float32)
                if gt[i, j] != 0:
                    temp[int(gt[i, j]) - 1] = 1
                GT_one_hot.append(temp)
        GT_one_hot = np.reshape(GT_one_hot, [gt.shape[0], gt.shape[1], class_count])
        return GT_one_hot

    def Lable_one_hot(self, label, class_count):
        one_hot = []
        for i in range(label.shape[0]):
            temp = np.zeros(class_count, dtype=np.float32)
            temp[int(label[i] - 1)] = 1
            one_hot.append(temp)
        one_hot = np.array(one_hot)
        return one_hot
    # 数据standardization标准化,即提前全局BN
    def Stand(self, data):
        height, width, bands = data.shape
        data = np.reshape(data, [height * width, bands])
        minMax = preprocessing.StandardScaler()
        data = minMax.fit_transform(data)
        data = np.reshape(data, [height, width, bands])
        return data


if __name__ == "__main__":
    HSI_data = HSI_dataset()
    original_data, gt, class_count, dataset_name, pca_components = HSI_data.data_select(1)
    height, width, bands = original_data.shape
