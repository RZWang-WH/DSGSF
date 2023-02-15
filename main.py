import numpy as np
import torch
import os
from utils import *
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from dataset import HSI_dataset
import random
from torch.utils.data import SubsetRandomSampler
from model.model import DSGSF


# 固定随机种子
seed = 2022
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)

# 数据集选择
HSI_data = HSI_dataset()
original_data, gt = HSI_data.data_select(2)
height, width, bands = original_data.shape
data = HSI_data.Stand(original_data)
learning_rate = HSI_data.learning_rate
batch_size = HSI_data.batch_size
# 数据标准化
PCA_data, num_components = Apply_PCA(original_data)
stand_data = PCA_data

# 原始数据
all_data = original_data.reshape([height * width, bands])
gt_reshape = np.reshape(gt, [-1])
# 获取采样下标
class_count = HSI_data.class_count
train_index, val_index, test_index, all_index = get_data_index(gt_reshape, HSI_data.train_ratio, HSI_data.val_ratio,
                                                               class_count)
stand_pre_data = stand_data.reshape([height * width, num_components])
print('num od train:', len(train_index))
print('num od val:', len(val_index))

train_data = all_data[train_index]
train_gt = np.array(get_sample_gt(train_index, gt_reshape), dtype=object)
val_gt = np.array(get_sample_gt(val_index, gt_reshape), dtype=object)
test_gt = np.array(get_sample_gt(test_index, gt_reshape), dtype=object)
# 独热编码
train_gt = np.reshape(train_gt, [height, width])
val_gt = np.reshape(val_gt, [height, width])
test_gt = np.reshape(test_gt, [height, width])

train_gt_onehot = HSI_data.GT_one_hot(train_gt, class_count).astype(int).reshape(height * width, -1)
test_gt_onehot = HSI_data.GT_one_hot(val_gt, class_count).astype(int).reshape(height * width, -1)
val_gt_onehot = HSI_data.GT_one_hot(test_gt, class_count).astype(int).reshape(height * width, -1)

all_gt_onehot = HSI_data.GT_one_hot(gt, class_count).astype(int).reshape(height * width, -1)

train_gt = np.reshape(train_gt, [height * width])
val_gt = np.reshape(val_gt, [height * width])
test_gt = np.reshape(test_gt, [height * width])

###转到GPU
train_data = torch.from_numpy(train_data.astype(np.float32)).cuda()
train_gt_onehot = torch.from_numpy(train_gt_onehot.astype(np.float32)).cuda()
val_gt_onehot = torch.from_numpy(val_gt_onehot.astype(np.float32)).cuda()
test_gt_onehot = torch.from_numpy(test_gt_onehot.astype(np.float32)).cuda()
all_gt_onehot = torch.from_numpy(all_gt_onehot.astype(np.float32)).cuda()

train_label = all_gt_onehot[train_index]
val_label = all_gt_onehot[val_index]
test_label = all_gt_onehot[test_index]

net_index = np.array(train_index, np.float32)
net_index = torch.from_numpy(net_index.astype(np.float32)).cuda()

all_net_index = np.array(all_index, np.float32)
all_net_index = torch.from_numpy(all_net_index.astype(np.float32)).cuda()
gc_input = np.array(stand_data, np.float32)
gc_input = torch.from_numpy(gc_input.astype(np.float32)).cuda()
net_input = np.array(all_data, np.float32)
net_input = torch.from_numpy(net_input.astype(np.float32)).cuda()

# 装载数据
trn_dataset = TensorDataset(train_data, train_label, net_index)
trn_loader = DataLoader(trn_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(range(train_data.shape[0])),
                        drop_last=True)

index = {'all_index': all_net_index,
         'train_index': train_index,
         'val_index': val_index,
         'test_index': test_index}
label = {'train': train_label,
         'val': val_label}
modelinput = {'gc': gc_input,
              'net': net_input}

model = DSGSF(bands, num_components, class_count)
model = model.cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-9, weight_decay=1e-5)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,100,200, 300], gamma=0.6)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=0)
OA_ALL = []
AA_ALL = []
KPP_ALL = []

AVG_ALL = []
# 记录损失值
Train_Loss = []
Val_Loss = []
best_loss = 99999
model.train()
#fig, ax = plt.subplots()
for i in range(HSI_data.max_epoch + 1):
    # optimizer.zero_grad()  # zero the gradient buffers
    for idx, (train, data_label, index) in enumerate(trn_loader):
        optimizer.zero_grad()
        output = model(train, gc_input, index)
        loss = criterion(output, data_label)
        loss.backward(retain_graph=False)
        optimizer.step()  # Does the update
    if i % 2 == 0:
        with torch.no_grad():
            model.eval()
            output = model(net_input, gc_input, all_net_index)
            trainloss = criterion(output[train_index], train_label)
            trainOA = compute_oa(output[train_index], train_label)
            valloss = criterion(output[val_index], val_label)
            valOA = compute_oa(output[val_index], val_label)
            print("{}\ttrain loss={}\t train OA={} val loss={}\t val OA={}".format(str(i + 1), trainloss, trainOA,
                                                                                   valloss, valOA))
            Train_Loss.append(trainloss.cpu().numpy())
            Val_Loss.append(valloss.cpu().numpy())
            if valloss < best_loss:
                best_loss = valloss
                torch.save(model.state_dict(), "./model/best_model.pt")
                print('save model...')
        torch.cuda.empty_cache()
        #classification_map = torch.argmax(output, 1).reshape([height, width]) + 1
        #wandbshow(classification_map.cpu().numpy(), class_count, fig,ax)
        model.train()
    # scheduler.step()
print("\n\n====================training done. starting evaluation...========================\n")
torch.cuda.empty_cache()
with torch.no_grad():
    model.load_state_dict(torch.load("./model/best_model.pt"))
    model.eval()
    output = model(net_input, gc_input, all_net_index)
    # testloss = compute_cross_loss(criterion, output[test_index], test_label)
    testloss = criterion(output[test_index], test_label)
    testOA = compute_oa(output[test_index], test_label)

    test_AA, test_AC_list = compute_aa(output[test_index], test_label, num_classes=class_count,
                                       return_accuracys=True)
    test_kpp = compute_kappa(output[test_index], test_label, num_classes=class_count)
    testOA = float(testOA.cpu())
    test_AA = float(test_AA.cpu())
    test_kpp = float(test_kpp.cpu())
    OA_ALL.append(testOA * 100)
    AA_ALL.append(test_AA * 100)
    KPP_ALL.append(test_kpp * 100)
    test_AC_list = [float(sub_list.cpu()) for sub_list in test_AC_list]
    AVG_ALL.append(test_AC_list)
    print("test OA=", testOA, "AA=", test_AA, 'kpp=', test_kpp)
    print('acc per class:')
    print(test_AC_list)
    # 计算
    classification_map = torch.argmax(output, 1).reshape([height, width]) + 1
    save_dir = os.path.join("./results", HSI_data.dataset_name)
    draw_loss(Train_Loss, Val_Loss, HSI_data.dataset_name)
    Draw_Classification_Map(classification_map.cpu().numpy(), gt, class_count,
                            save_dir + "/" + HSI_data.dataset_name + str(testOA))
torch.cuda.empty_cache()

OA_ALL = np.array(OA_ALL)
AA_ALL = np.array(AA_ALL)
KPP_ALL = np.array(KPP_ALL)
AVG_ALL = np.array(AVG_ALL)
print("\ntrain_ratio={}\nlearn_rate={}".format(HSI_data.train_ratio, HSI_data.learning_rate),
      "\n==============================================================================")
print('OA=', np.mean(OA_ALL), '+-', np.std(OA_ALL))
print('AA=', np.mean(AA_ALL), '+-', np.std(AA_ALL))
print('Kpp=', np.mean(KPP_ALL), '+-', np.std(KPP_ALL))
print('AVG=', np.mean(AVG_ALL, 0), '+-', np.std(AVG_ALL, 0))
Draw_Classification_Map(gt, gt, class_count, save_dir + "/" + HSI_data.dataset_name)
