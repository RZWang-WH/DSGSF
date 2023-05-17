import numpy as np
import torch
import matplotlib.pyplot as plt


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_loss(prediction, label):
    we = -torch.mul(label, torch.log(prediction))
    cross_entropy = torch.sum(we)
    return cross_entropy
'''
def compute_loss(criterion, predict: torch.Tensor, label: torch.Tensor):
    loss = 0.0
    loss = criterion(predict, label)
    return loss
# criterion = torch.nn.CrossEntropyLoss()
'''
def compute_cross_loss(criterion, predict, label):
    loss = criterion(predict, label)
    return loss


def confusion_matrix(pred: torch.Tensor, label: torch.Tensor, num_classes=None):
    size = [num_classes , num_classes ] if num_classes is not None else None
    y_true = label.float()
    y_pred = pred.float()
    if size is None:
        cm = torch.sparse_coo_tensor(indices=torch.stack([y_true, y_pred], dim=0), values=torch.ones_like(y_pred))
    else:
        cm = torch.sparse_coo_tensor(indices=torch.stack([y_true, y_pred], dim=0), values=torch.ones_like(y_pred),
                                     size=size)
    return cm.to_dense()


def compute_oa(network_output, samples_gt_onehot):
    return (torch.argmax(network_output, 1) == torch.argmax(samples_gt_onehot, 1)).sum().float() / float(
        samples_gt_onehot.shape[0])


def compute_aa(network_output, samples_gt_onehot, num_classes=None, return_accuracys=False):

    cm_th = confusion_matrix(torch.argmax(samples_gt_onehot,1), torch.argmax(network_output,1), num_classes)
    cm_th = cm_th.float()

    aas = torch.diag(cm_th / (cm_th.sum(dim=1)))
    if not return_accuracys:
        return aas.mean()
    else:
        return aas.mean(), aas
'''
def compute_aa(network_output, samples_gt_onehot, num_classes=None, return_accuracys=False):
    zero_vector = np.zeros([num_classes])
    output_data = network_output.cpu().numpy()
    samples_gt_onehot = samples_gt_onehot.cpu().numpy()
    output_data = np.reshape(output_data, [-1, num_classes])
    idx = np.argmax(output_data, axis=-1)  # 判别类别
    for z in range(output_data.shape[0]):
        if ~(zero_vector == output_data[z]).all():
            idx[z] += 1
    count_perclass = np.zeros([num_classes])
    correct_perclass = np.zeros([num_classes])
    for x in range(len(samples_gt_onehot)):
        if np.argmax(samples_gt_onehot[x]) != 0:
            count_perclass[np.argmax(samples_gt_onehot[x]) - 1] += 1
            if np.argmax(samples_gt_onehot[x]) == idx[x]:
                correct_perclass[np.argmax(samples_gt_onehot[x]) - 1] += 1
    test_AC_list = correct_perclass / count_perclass
    test_AA = np.average(test_AC_list)
    if return_accuracys==False:
        return test_AA
    else:
        return test_AA,test_AC_list
'''

def compute_kappa(network_output, samples_gt_onehot, num_classes=None):
    cm_th = confusion_matrix(torch.argmax(samples_gt_onehot, 1), torch.argmax(network_output, 1), num_classes)
    cm_th = cm_th.float()
    n_classes = cm_th.size(0)
    sum0 = cm_th.sum(dim=0)
    sum1 = cm_th.sum(dim=1)
    expected = torch.ger(sum0, sum1) / torch.sum(sum0)
    w_mat = torch.ones([n_classes, n_classes], dtype=torch.float32).cuda()
    w_mat.view(-1)[:: n_classes + 1] = 0.
    k = torch.sum(w_mat * cm_th) / torch.sum(w_mat * expected)
    return 1. - k


def draw_loss(Train_Loss, Val_Loss,dataset_name):
    # 设置图例并且设置图例的字体及大小
    font1 = {
             'weight': 'normal',
             'size': 15,
             }
    # 绘制损失函数曲线
    plt.figure(0)
    plt.tick_params(labelsize=12)
    # history_dict = history_fdssc.history
    loss_value = np.array(Train_Loss)[1:]
    val_loss_value = np.array(Val_Loss)[1:]
    epochs = range(1, loss_value.size + 1) * np.array(5, dtype=np.int)
    plt.plot(epochs, loss_value, "bo", label="Training loss", linewidth=3, markersize=5)
    plt.plot(epochs, val_loss_value, "r", label="Validation loss", linewidth=3)
    plt.xlabel("Iterations", font1)
    plt.ylabel("Loss", font1)
    plt.legend()
    plt.show()
    plt.savefig('./results//' + dataset_name + '_LOSS', dpi=400)  # 指定分辨率
    plt.close()
