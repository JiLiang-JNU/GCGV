import math
import numpy as np

def Data_Split(gt_reshape, class_num, train_ratio, val_ratio, train_num1,train_num2, val_num, samples_type):
    train_index = []
    test_index = []
    val_index = []
    if samples_type == 'ratio':
        for i in range(class_num):
            idx = np.where(gt_reshape == i + 1)[-1]
            samplesCount = len(idx)
            # print("Class ",i,":", samplesCount)
            max_index = np.max(samplesCount) + 1
            np.random.shuffle(idx)
            if samplesCount <= 60:
                sample_num = math.ceil(samplesCount * train_ratio)
            else:
                sample_num = math.ceil(samplesCount * train_ratio)
            #print("Class ", i, ":", sample_num)  # 每一类训练的个数
            # 取出每个类别选择出的训练集
            train_index.append(idx[: sample_num])
            val_index.append(idx[sample_num: sample_num + val_num])
            test_index.append(idx[sample_num + val_num:])

    else:
        sample_num = train_num1
        for i in range(class_num):
            idx = np.where(gt_reshape == i + 1)[-1]
            samplesCount = len(idx)
            #print("Class ",i,":", samplesCount)
            max_index = np.max(samplesCount) + 1
            np.random.shuffle(idx)
            if samplesCount <= 60:
                sample_num = train_num1
            else:
                sample_num = train_num2
            #print("Class ", i, ":", sample_num)  # 每一类训练的个数
            train_index.append(idx[: sample_num])
            val_index.append(idx[sample_num : sample_num+val_num])
            test_index.append(idx[sample_num+val_num : ])

    train_index = np.concatenate(train_index, axis=0)
    val_index = np.concatenate(val_index, axis=0)
    test_index = np.concatenate(test_index, axis=0)
    return train_index, val_index, test_index

