import numpy as np
import math
import matplotlib.pyplot as plt
import spectral as spy
from matplotlib import cm
from sklearn.decomposition import PCA


def get_data_index(groundtruth_reshape, train_ratio, val_ratio, class_num,sample='propor',ratio =0.5):
    data_num = []
    train_rand_idx = []
    for i in range(class_num):
        idx = np.where(groundtruth_reshape == i + 1)[-1]
        data_num.append(len(idx))
    if sample == 'universe':
        sample_num = [max(math.ceil(i * train_ratio), 10) for i in data_num]
        val_sample = [max(math.ceil(i * val_ratio), 2) for i in data_num]
    elif sample == 'propor':
        sample_num = [max(math.ceil(i * train_ratio), 5) // ratio for i in data_num]
    #sample_num = [max(math.ceil(i * train_ratio), 5) for i in data_num]
    print('train:',sample_num)
    print('val',val_sample)
    for i in range(class_num):
        idx = np.where(groundtruth_reshape == i + 1)[-1]
        rand_list = [i for i in range(len(idx))]  # 对应类数目的列表
        rand_idx = np.random.choice(rand_list, int(sample_num[i]), replace=False)  # 随机数数量 四舍五入(改为上取整)
        rand_real_idx_per_class = idx[rand_idx]  # 随机抽取的样本存入列表
        train_rand_idx.append(rand_real_idx_per_class)
    train_rand_idx = np.array(train_rand_idx, dtype=object)

    # 数据划分
    all_data_index = [i for i in range(len(groundtruth_reshape))]
    train_data_index = []
    for c in range(train_rand_idx.shape[0]):
        a = train_rand_idx[c]
        for j in range(a.shape[0]):
            train_data_index.append(a[j])
    train_data_index = np.array(train_data_index, dtype=object)

    val_rand_idx = []
    for i in range(class_num):
        idx = np.where(groundtruth_reshape == i + 1)[-1]
        rand_list = [j for j in range(len(set(idx)-set(train_rand_idx[i])))]  # 对应类数目的列表
        rand_idx = np.random.choice(rand_list, int(val_sample[i]), replace=False)  # 随机数数量 四舍五入(改为上取整)
        rand_real_idx_per_class = idx[rand_idx]  # 随机抽取的样本存入列表
        val_rand_idx.append(rand_real_idx_per_class)
    val_rand_idx = np.array(val_rand_idx, dtype=object)
    val_data_index = []
    for c in range(val_rand_idx.shape[0]):
        a = val_rand_idx[c]
        for j in range(a.shape[0]):
            val_data_index.append(a[j])
    val_data_index = np.array(val_data_index, dtype=object)

    background_idx = np.where(groundtruth_reshape == 0)[-1]  # 背景像素获取
    all_data_index = set(all_data_index)
    background_idx = set(background_idx)
    train_data_index = set(train_data_index)
    val_data_index = set(val_data_index)
    test_data_index = all_data_index - train_data_index - background_idx-val_data_index  # 测试集为总数据减去训练集
    # val_data_count = int(math.ceil(val_ratio * (len(test_data_index) + len(train_data_index))))  # 验证集数量
    # test_data_index = list(test_data_index)
    # val_data_index = np.random.choice(test_data_index, val_data_count, replace=False)  # 验证集从测试集抽取(不重复采样)
    # val_data_index = set(val_data_index)
    # test_data_index = set(test_data_index)
    # test_data_index -= val_data_index  # 最终测试集为原测试集减去验证集

    test_data_index = list(test_data_index)
    train_data_index = list(train_data_index)
    val_data_index = list(val_data_index)
    all_data_index = list(all_data_index)
    return train_data_index, val_data_index, test_data_index, all_data_index


def get_sample_gt(data_index, gt_reshape):
    sample_gt = np.zeros(gt_reshape.shape)
    for i in range(len(data_index)):
        sample_gt[data_index[i]] = gt_reshape[data_index[i]]
    return sample_gt


def get_data_mask(sample_gt, height, width, class_count):
    label_mask = np.zeros([height * width, class_count])
    temp_ones = np.ones([class_count])
    sample_gt = np.reshape(sample_gt, [height * width])
    for i in range(height * width):
        if sample_gt[i] != 0:
            label_mask[i] = temp_ones
    label_mask = np.reshape(label_mask, [height * width, class_count])
    return label_mask

#标签独热编码
def Label_one_hot(label, class_count):
    one_hot = []
    for i in range(label.shape[0]):
        temp = np.zeros(class_count, dtype=np.float32)
        if label[i] != 0:
            temp[int(label[i]) - 1] = 1
        one_hot.append(temp)
    one_hot = np.reshape(one_hot, [label.shape[0], class_count])
    return one_hot


# PCA降维
def Apply_PCA(data):
    # PCA主成分分析
    pc = spy.principal_components(data)
    pc_98 = pc.reduce(fraction=0.98)  # 保留98%的特征值
    print("PCA_DIM",len(pc_98.eigenvalues))  # 剩下的特征值数量
    num_components = len(pc_98.eigenvalues)
    #spy.imshow(data=pc.cov, title="pc_cov")
    img_pc = pc_98.transform(data)  # 把数据转换到主成分空间
    #spy.imshow(img_pc[:, :, :3], stretch_all=True)  # 前三个主成分显示
    return img_pc,num_components

    # new_data = np.reshape(data, (-1, data.shape[2]))
    # pca = PCA(n_components, whiten=True)
    # new_data = pca.fit_transform(new_data)
    # new_data = np.reshape(new_data, (data.shape[0], data.shape[1], n_components))
    # return new_data,n_components


# 数据边缘填充
def Padding(data, margin=2):
    new_data = np.zeros((data.shape[0] + 2 * margin, data.shape[1] + 2 * margin, data.shape[2]))
    x_offset = margin
    y_offset = margin
    new_data[x_offset:data.shape[0] + x_offset, y_offset:data.shape[1] + y_offset, :] = data
    return new_data


# 根据index创建训练样本
def Data_Gen(index, re_data, label, WindowSize):
    margin = int((WindowSize - 1) / 2)
    all_padddle = Padding(re_data, margin=margin)
    gen_data = np.zeros((len(index), WindowSize, WindowSize, re_data.shape[2]))
    gen_labels = np.zeros(len(index))
    for i in range(len(index)):
        r = index[i] // re_data.shape[1]  # 求行
        c = index[i] % re_data.shape[1]  # 求列
        gen_data[i, :, :, :] = all_padddle[r:r + (2 * margin) + 1, c:c + (2 * margin) + 1]
        gen_labels[i] = label[index[i]]
    return gen_data, gen_labels


def Draw_Classification_Map(label, gt, class_count, name: str, scale: float = 4.0, dpi: int = 400):
    height, width = label.shape
    cmap = cm.get_cmap('jet', class_count + 1)
    plt.set_cmap(cmap)
    temp_zeros = np.zeros((height, width))
    fig, ax = plt.subplots()
    truth = np.where(gt != 0, label, temp_zeros)
    v = spy.imshow(classes=truth.astype(np.int16), fignum=fig.number)
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    foo_fig = plt.gcf()  # 'get current figure'
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    foo_fig.show()
    foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
    # out = mark_boundaries(fig[:,:,[0,1,2]], segments)
    # out.savefig('segments.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
    pass
