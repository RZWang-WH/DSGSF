import numpy as np
import torch
import os
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from dataset import HSI_dataset
from utils import get_data_index, get_sample_gt, Padding, Apply_PCA, Data_Gen, Label_one_hot, Draw_Classification_Map
from torch.utils.data import SubsetRandomSampler
from model import U_GC_LSTM
from function import compute_loss, compute_cross_loss, draw_loss, compute_oa, compute_aa, compute_kappa
import random

# 参数设置
batch_size = 16
weight_decay = 1e-5

seed = 2022
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 数据集选择
HSI_data = HSI_dataset()
original_data, gt, class_count, dataset_name = HSI_data.data_select(1)
height, width, bands = original_data.shape
original_data = HSI_data.Stand(original_data)
# 数据标准化
# PCA_data, num_components = Apply_PCA(original_data)
PCA_data = np.load('./pca_data/'+dataset_name+'.npy')
num_components = PCA_data.shape[-1]
stand_data = HSI_data.Stand(PCA_data)
# [h,w,num_components]
# 原始数据
all_data = original_data.reshape([height * width, bands])
gt_reshape = np.reshape(gt, [-1])

# 获取采样下标
train_index, val_index, test_index, all_index = get_data_index(gt_reshape, HSI_data.train_ratio, HSI_data.val_ratio,
                                                               class_count, HSI_data.sample, HSI_data.sample_ratio)
stand_pre_data = stand_data.reshape([height * width, num_components])

# %%
# train_data = stand_pre_data[train_index]
train_data = all_data[train_index]
test_data = stand_pre_data[test_index]
val_data = stand_pre_data[val_index]

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
val_data = torch.from_numpy(val_data.astype(np.float32)).cuda()
test_data = torch.from_numpy(test_data.astype(np.float32)).cuda()

# train_gt = torch.from_numpy(train_gt.astype(np.float32)).cuda()
# val_gt = torch.from_numpy(val_gt.astype(np.float32)).cuda()
# test_gt = torch.from_numpy(test_gt.astype(np.float32)).cuda()

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

##装载数据
print(train_data.size())
trn_dataset = TensorDataset(train_data, train_label, net_index)
trn_loader = DataLoader(trn_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(range(train_data.shape[0])),
                        drop_last=True)

print("data load ok!")

model = U_GC_LSTM(bands, num_components, class_count)
model = model.cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=HSI_data.learning_rate, betas=(0.9, 0.999), eps=1e-9,
                             weight_decay=weight_decay)
# optimizer = torch.optim.SGD(model.parameters(), lr=HSI_data.learning_rate)

# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,100,200, 300], gamma=0.6)
print("set ok")
# 记录损失值
Train_Loss = []
Val_Loss = []
OA_ALL = []
AA_ALL = []
KPP_ALL = []
AVG_ALL = []
best_loss = 99999
model.train()
for i in range(HSI_data.max_epoch + 1):
    optimizer.zero_grad()  # zero the gradient buffers
    for idx, (train, data_label, index) in enumerate(trn_loader):
        output = model(train, gc_input, index)
        # for j in range(len(output)):
        #     if j == 0:
        #         loss = criterion(output[j], data_label)
        #     if j > 0:
        #         loss += criterion(output[j], data_label)
        # loss = compute_cross_loss(criterion, output[index.tolist()], data_label)
        loss = criterion(output, data_label)
        optimizer.zero_grad()
        loss.backward(retain_graph=False)
        optimizer.step()  # Does the update
    if i % 5 == 0:
        with torch.no_grad():
            model.eval()
            output = model(net_input, gc_input, all_net_index)
            # trainloss = compute_cross_loss(criterion, output[train_index], train_label)
            # print(output[train_index].size())
            # print(train_label.size())
            trainloss = criterion(output[train_index], train_label)
            trainOA = compute_oa(output[train_index], train_label)
            # valloss = compute_cross_loss(criterion, output[val_index], val_label)
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

    test_AA, test_AC_list = compute_aa(output[test_index], test_label, num_classes=class_count, return_accuracys=True)
    test_kpp = compute_kappa(output[test_index], test_label, num_classes=class_count)
    # testOA = evaluate_performance(OA_ALL, AA_ALL, KPP_ALL, AVG_ALL, class_count, Test_GT, output, test_gt,test_gt_onehot, require_AA_KPP=True, printFlag=False)
    testOA = float(testOA.cpu())
    test_AA = float(test_AA.cpu())
    test_kpp = float(test_kpp.cpu())
    OA_ALL.append(testOA)
    AA_ALL.append(test_AA)
    KPP_ALL.append(test_kpp)
    test_AC_list = [float(sub_list.cpu()) for sub_list in test_AC_list]
    AVG_ALL.append(test_AC_list)
    # 保存数据信息
    f = open('results/' + dataset_name + '/' + dataset_name + '_results.txt', 'a+')
    str_results = '\n======================' \
                  + " learning rate=" + str(HSI_data.learning_rate) \
                  + " epochs=" + str(HSI_data.max_epoch) \
                  + " train ratio=" + str(HSI_data.train_ratio) \
                  + " val ratio=" + str(HSI_data.val_ratio) \
                  + " ======================" \
                  + "\nOA=" + str(testOA) \
                  + "\nAA=" + str(test_AA) \
                  + '\nkpp=' + str(test_kpp) \
                  + '\nacc per class:' + str(test_AC_list) + "\n"
    # + '\ntrain time:' + str(time_train_end - time_train_start) \
    # + '\ntest time:' + str(time_test_end - time_test_start) \
    f.write(str_results)
    f.close()
    print("test OA=", testOA, "AA=", test_AA, 'kpp=', test_kpp)
    print('acc per class:')
    print(test_AC_list)
    # 计算
    classification_map = torch.argmax(output, 1).reshape([height, width]) + 1
    save_dir = os.path.join("./results", dataset_name)
    Draw_Classification_Map(classification_map.cpu().numpy(), gt, class_count,
                            save_dir + "/" + dataset_name + str(testOA))
torch.cuda.empty_cache()
del model

OA_ALL = np.array(OA_ALL)
AA_ALL = np.array(AA_ALL)
KPP_ALL = np.array(KPP_ALL)
AVG_ALL = np.array(AVG_ALL)
print("\ntrain_ratio={}".format(HSI_data.train_ratio),
      "\n==============================================================================")
print('OA=', np.mean(OA_ALL), '+-', np.std(OA_ALL))
print('AA=', np.mean(AA_ALL), '+-', np.std(AA_ALL))
print('Kpp=', np.mean(KPP_ALL), '+-', np.std(KPP_ALL))
print('AVG=', np.mean(AVG_ALL, 0), '+-', np.std(AVG_ALL, 0))
# print("Average training time:{}".format(np.mean(Train_Time_ALL)))
# print("Average testing time:{}".format(np.mean(Test_Time_ALL)))
# 绘制损失曲线图
draw_loss(Train_Loss, Val_Loss, dataset_name)
Draw_Classification_Map(gt, gt, class_count, save_dir + "/" + dataset_name)
