from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import numpy as np
import scipy.io as sio
import torch
import math
import matplotlib.pyplot as plt
import yaml

f = open(r'/home/JiLiang/GCGV/config/config.yaml', 'r', encoding='utf-8')
result = f.read()
config = yaml.load(result, Loader=yaml.FullLoader)

train_com = config["data_split"]["train_num"]
train_num1 = config["data_split"]["train_num1"]
train_num2 = config["data_split"]["train_num2"]
samples_type = config["data_split"]["samples_type"]
train_ratio = config["data_split"]["train_ratio"]
dataset_name = config["data_input"]["dataset_name"]


def Top(image, cornor_index, x, y, patch, b, n_top):
    sort = image.reshape(patch * patch, b)
    sort = torch.from_numpy(sort).type(torch.FloatTensor)
    pos = (x - cornor_index[0]) * patch + (y - cornor_index[1])
    Q = torch.sum(torch.pow(sort[pos] - sort, 2), dim=1)
    _, indices = Q.topk(k=n_top, dim=0, largest=False, sorted=True)
    return indices


def Patch(pca_image, point, i, patch, W, H, n_gcn):
    x = point[i, 0]
    y = point[i, 1]
    m = int((patch - 1) / 2)
    _, _, b = pca_image.shape
    if x <= m:
        if y <= m:
            temp_image = pca_image[0:patch, 0:patch, :]
            cornor_index = [0, 0]
        if y >= (H - m):
            temp_image = pca_image[0:patch, H - patch:H, :]
            cornor_index = [0, H - patch]
        if y > m and y < H - m:
            temp_image = pca_image[0:patch, y - m:y + m + 1, :]
            cornor_index = [0, y - m]
    if x >= (W - m):
        if y <= m:
            temp_image = pca_image[W - patch:W, 0:patch, :]
            cornor_index = [W - patch, 0]
        if y >= (H - m):
            temp_image = pca_image[W - patch:W, H - patch:H, :]
            cornor_index = [W - patch, H - patch]
        if y > m and y < H - m:
            temp_image = pca_image[W - patch:W, y - m:y + m + 1, :]
            cornor_index = [W - patch, y - m]
    if x > m and x < W - m:
        if y <= m:
            temp_image = pca_image[x - m:x + m + 1, 0:patch, :]
            cornor_index = [x - m, 0]
        if y >= (H - m):
            temp_image = pca_image[x - m:x + m + 1, H - patch:H, :]
            cornor_index = [x - m, H - patch]
        if y > m and y < H - m:
            temp_image = pca_image[x - m:x + m + 1, y - m:y + m + 1, :]
            cornor_index = [x - m, y - m]

    index = Top(temp_image, cornor_index, x, y, patch, b, n_gcn)
    return temp_image, cornor_index, index


def Train_Test_Data(pca_image, band, train_point, test_point, true_point, patch, w, h, n_gcn):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)
    x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=float)
    corner_train = np.zeros((train_point.shape[0], 2), dtype=int)
    corner_test = np.zeros((test_point.shape[0], 2), dtype=int)
    corner_true = np.zeros((true_point.shape[0], 2), dtype=int)
    indexs_train = torch.zeros((train_point.shape[0], n_gcn), dtype=int).cuda()
    indexs_test = torch.zeros((test_point.shape[0], n_gcn), dtype=int).cuda()
    indexs_ture = torch.zeros((true_point.shape[0], n_gcn), dtype=int).cuda()
    for i in range(train_point.shape[0]):
        x_train[i, :, :, :], corner_train[i, :], indexs_train[i] = Patch(pca_image, train_point, i,
                                                                                           patch, w, h, n_gcn)
    for j in range(test_point.shape[0]):
        x_test[j, :, :, :], corner_test[j, :], indexs_test[j] = Patch(pca_image, test_point, j, patch,
                                                                                        w, h, n_gcn)
    for k in range(true_point.shape[0]):
        x_true[k, :, :, :], corner_true[k, :], indexs_ture[k] = Patch(pca_image, true_point, k, patch,
                                                                                        w, h, n_gcn)
    print("x_train shape = {}, type = {}".format(x_train.shape, x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape, x_test.dtype))
    print("x_true  shape = {}, type = {}".format(x_true.shape, x_test.dtype))
    print("**************************************************")

    return x_train, x_test, x_true, corner_train, corner_test, corner_true, indexs_train, indexs_test, indexs_ture


class Meter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def Accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, target, pred.squeeze()


def Train(GCN, Transformer, Branch1, Branch1_in, train_loader, criterion, optimizer1, optimizer2, optimizer3,
                indexs_train, branch1, branch2):
    objs = Meter()
    top1 = Meter()
    tar = np.array([])
    pre = np.array([])
    poses = np.array([])
    pos_new = []
    for batch_idx, (A, batch_data, batch_target, pos) in enumerate(train_loader):
        batch_A = A.cuda()
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()
        pos = pos.cuda()
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        GCN_Pre = GCN(batch_data, batch_A, indexs_train)
        Transformer_Pre = Transformer(GCN_Pre)
        # print("pos", pos.shape)
        # print("Branch1_in.shape",Branch1_in.shape)
        Branch1_Pre = Branch1(Branch1_in)


        if dataset_name == "IndianPines":
            Branch1_Pre = Branch1_Pre.reshape((-1, 145, 16)) #IP数据集

        elif dataset_name == "paviaU":
            Branch1_Pre = Branch1_Pre.reshape((-1, 340, 9))  # UP数据集

        elif dataset_name == "Salinas":
            Branch1_Pre = Branch1_Pre.reshape((-1, 217, 16))  # SA数据集

        elif dataset_name == "pavia":
            Branch1_Pre = Branch1_Pre.reshape((-1, 715, 9))  # pavia数据集

        elif dataset_name == "KSC":
            Branch1_Pre = Branch1_Pre.reshape((-1, 614, 13))  # ksc数据集

        # print("Branch1_Pre.shape",Branch1_Pre.shape)
        # print("Branch1_Pre.shape",Branch1_Pre.shape)

        temp = torch.empty(0).cuda()

        for i in range(len(pos)):
            x = pos[i][0]
            y = pos[i][1]
            temp = torch.cat((temp, Branch1_Pre[x][y].unsqueeze(0)))
        # print("temp.shape",temp.shape)
        # print("Transformer_Pre.shape",Transformer_Pre.shape)
        #权重融合
        batch_pred = branch1 * Transformer_Pre + branch2 * temp

        # print(Transformer_Pre)

        # print(batch_pred.shape)torch.Size([64, 16])
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer1.step()
        optimizer2.step()
        optimizer3.step()
        prec1, t, p = Accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
        pos_new.extend(pos.data.cpu().numpy().tolist())
        poses = np.append(poses, pos.data.cpu().numpy())

    pos_new = np.array(pos_new).reshape(-1, 2)
    # print(tar.shape)
    # print(pre.shape)
    # print(pos_new.shape) #[435,2]
    return top1.avg, objs.avg, tar, pre, poses, pos_new


def Valid(GCN, Transformer, Branch1, Branch1_in, valid_loader, criterion, indexs_test, branch1, branch2):
    objs = Meter()
    top1 = Meter()
    tar = np.array([])
    pre = np.array([])

    pos_new = []

    for batch_idx, (A, batch_data, batch_target, pos) in enumerate(valid_loader):
        batch_A = A.cuda()
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()
        GCN_Pre = GCN(batch_data, batch_A, indexs_test)
        Transformer_Pre = Transformer(GCN_Pre)
        Branch1_Pre = Branch1(Branch1_in)

        if dataset_name == "IndianPines":
            Branch1_Pre = Branch1_Pre.reshape((-1, 145, 16)) #IP数据集

        elif dataset_name == "paviaU":
            Branch1_Pre = Branch1_Pre.reshape((-1, 340, 9))  # UP数据集

        elif dataset_name == "Salinas":
            Branch1_Pre = Branch1_Pre.reshape((-1, 217, 16))  # SA数据集

        elif dataset_name == "pavia":
            Branch1_Pre = Branch1_Pre.reshape((-1, 715, 9))  # pavia数据集

        elif dataset_name == "KSC":
            Branch1_Pre = Branch1_Pre.reshape((-1, 614, 13))  # ksc数据集

        temp = torch.empty(0).cuda()
        for i in range(len(pos)):
            x = pos[i][0]
            y = pos[i][1]
            temp = torch.cat((temp, Branch1_Pre[x][y].unsqueeze(0)))
        batch_pred = branch1 * Transformer_Pre + branch2 * temp
        loss = criterion(Transformer_Pre, batch_target)
        prec1, t, p = Accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
        pos_new.extend(pos.data.cpu().numpy().tolist())

    pos_new = np.array(pos_new).reshape(-1, 2)
    return tar, pre, pos_new


def Metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = Result_output(matrix)
    # print(pre.shape)
    return OA, AA_mean, Kappa, AA


def Result_output(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float64)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA


def PCA_Process(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


def A_Process(temp_image, input2, corner, patches, l, sigma=10, ):
    input2 = input2.cuda()
    N, h, w, _ = temp_image.shape
    B = np.zeros((w * h, w * h), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            m = int(i * w + j)
            for k in range(l):
                for q in range(l):
                    n = int((i + (k - (l - 1) / 2)) * w + (j + (q - (l - 1) / 2)))
                    if 0 <= i + (k - (l - 1) / 2) < h and 0 <= (j + (q - (l - 1) / 2)) < w and m != n:
                        B[m, n] = 1

    index = np.argwhere(B == 1)
    index2 = np.where(B == 1)
    A = np.zeros((N, w * h, w * h), dtype=np.float32)

    for i in range(N):
        C = np.array(B)
        x_l = int(corner[i, 0])
        x_r = int(corner[i, 0] + patches)
        y_l = int(corner[i, 1])
        y_r = int(corner[i, 1] + patches)
        D = Corner(input2[x_l:x_r, y_l:y_r, :], sigma)
        D = D.cpu().numpy()
        C[index2[0], index2[1]] = D[index2[0], index2[1]]
        A[i, :, :] = C
    A = torch.from_numpy(A).type(torch.FloatTensor).cuda()
    return A


def Corner(A, sigma=10):
    height, width, band = A.shape
    A = A.reshape(height * width, band)
    prod = torch.mm(A, A.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod)
    res = (norm + norm.t() - 2 * prod).clamp(min=0)
    D = torch.exp(-res / (sigma ** 2))
    return D


def Normalize(input):
    input_normalize = np.zeros(input.shape)
    for i in range(input.shape[2]):
        input_max = np.max(input[:, :, i])
        input_min = np.min(input[:, :, i])
        input_normalize[:, :, i] = (input[:, :, i] - input_min) / (input_max - input_min)
    return input_normalize


def load_dataset(Dataset):
    if Dataset == 'Indian':
        mat_data = sio.loadmat('/home/JiLiang/Datasets/HSI-Datasets/Indian_pines_corrected.mat')
        mat_gt = sio.loadmat('/home/JiLiang/Datasets/HSI-Datasets/Indian_pines_gt.mat')
        data_hsi = mat_data["data"]
        gt_hsi = mat_gt["groundT"]
        TOTAL_SIZE = 10249
        VALIDATION_SPLIT = 0.97
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'PaviaU':
        uPavia = sio.loadmat('/home/JiLiang/Datasets/HSI-Datasets/PaviaU.mat')
        gt_uPavia = sio.loadmat('/home/JiLiang/Datasets/HSI-Datasets/PaviaU_gt.mat')
        data_hsi = uPavia['paviaU']
        gt_hsi = gt_uPavia['paviaU_gt']
        TOTAL_SIZE = 42776
        VALIDATION_SPLIT = 0.995
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'Salinas':
        SV = sio.loadmat('/home/JiLiang/Datasets/HSI-Datasets/Salinas_corrected.mat')
        gt_SV = sio.loadmat('/home/JiLiang/Datasets/HSI-Datasets/Salinas_gt.mat')
        data_hsi = SV['salinas_corrected']
        gt_hsi = gt_SV['salinas_gt']
        TOTAL_SIZE = 54129
        VALIDATION_SPLIT = 0.995
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'Pavia':
        uPavia = sio.loadmat('/home/JiLiang/Datasets/HSI-Datasets/Pavia.mat')
        gt_uPavia = sio.loadmat('/home/JiLiang/Datasets/HSI-Datasets/Pavia_gt.mat')
        data_hsi = uPavia['pavia']
        gt_hsi = gt_uPavia['pavia_gt']
        TOTAL_SIZE = 148152
        VALIDATION_SPLIT = 0.995
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'KSC':
        uPavia = sio.loadmat('/home/JiLiang/Datasets/HSI-Datasets/KSC.mat')
        gt_uPavia = sio.loadmat('/home/JiLiang/Datasets/HSI-Datasets/KSC_gt.mat')
        data_hsi = uPavia['KSC']
        gt_hsi = gt_uPavia['KSC_gt']
        TOTAL_SIZE = 5211
        VALIDATION_SPLIT = 0.995
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)


    return data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT


def sampling(proportion, ground_truth, CLASSES_NUM):
    train = {}
    test = {}
    train_num = []
    test_num = []
    labels_loc = {}
    for i in range(CLASSES_NUM):
        indexes = np.argwhere(ground_truth == (i + 1))
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        if proportion != 1:
            if samples_type == "number":  # 固定值训练
                nb_val = max(int((1 - proportion) * len(indexes)), 3)
                if indexes.shape[0] <= train_com:
                    nb_val = train_num1
                else:
                    nb_val = train_num2
            elif samples_type == "ratio":  # 比例ratio训练
                nb_val = max(int((1 - proportion) * len(indexes)), 3)
                if indexes.shape[0] <= train_com:
                    nb_val = np.ceil(indexes.shape[0] * train_ratio).astype('int32')
                else:
                    nb_val = np.ceil(indexes.shape[0] * train_ratio).astype('int32')

            print("Class ", i, ":", str(nb_val))  # 每一类训练的个数

        else:
            nb_val = 0

        train_num.append(nb_val)
        test_num.append(len(indexes) - nb_val)
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes = train[0]
    test_indexes = test[0]
    for i in range(CLASSES_NUM - 1):
        train_indexes = np.concatenate((train_indexes, train[i + 1]), axis=0)
        test_indexes = np.concatenate((test_indexes, test[i + 1]), axis=0)
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    # print(train)
    print(train_indexes.shape, test_indexes.shape, train_num, test_num)
    return train_indexes, test_indexes, train_num, test_num, train, test




def label(indices, gt_hsi):
    dim_0 = indices[:, 0]
    dim_1 = indices[:, 1]
    label = gt_hsi[dim_0, dim_1]
    return label


def get_data(dataset):
    data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT = load_dataset(dataset)
    gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]), )
    CLASSES_NUM = max(gt)
    train_indices, test_indices, train_num, test_num, train_index, test_index = sampling(VALIDATION_SPLIT, gt_hsi,
                                                                                         CLASSES_NUM)
    _, total_indices, _, total_num, total_index, total_index = sampling(1, gt_hsi, CLASSES_NUM)
    y_train = label(train_indices, gt_hsi) - 1
    y_test = label(test_indices, gt_hsi) - 1
    y_true = label(total_indices, gt_hsi) - 1
    # print(total_index)
    return data_hsi, CLASSES_NUM, train_indices, test_indices, total_indices, y_train, y_test, y_true


def Metrics(best_OA2, best_AA_mean2, best_Kappa2, AA2):
    results = {}
    results["OA"] = best_OA2 * 100.0
    results['AA'] = best_AA_mean2 * 100.0
    results["Kappa"] = best_Kappa2 * 100.0
    results["class acc"] = AA2 * 100.0
    return results


def Result_Show(results, agregated=False):
    text = ""
    if agregated:
        accuracies = [r["OA"] for r in results]
        aa = [r['AA'] for r in results]
        kappas = [r["Kappa"] for r in results]
        class_acc = [r["class acc"] for r in results]
        class_acc_mean = np.mean(class_acc, axis=0)
        class_acc_std = np.std(class_acc, axis=0)

    else:
        Accuracy = results["OA"]
        aa = results['AA']
        classacc = results["class acc"]
        kappa = results["Kappa"]

    text += "---\n"
    text += "class acc :\n"
    if agregated:
        for score, std in zip(class_acc_mean,
                              class_acc_std):
            text += "\t{:.02f} +- {:.02f}\n".format(score, std)
    else:
        for score in classacc:
            text += "\t {:.02f}\n".format(score)
    text += "---\n"

    if agregated:
        text += ("OA: {:.02f} +- {:.02f}\n".format(np.mean(accuracies),
                                                   np.std(accuracies)))
        text += ("AA: {:.02f} +- {:.02f}\n".format(np.mean(aa),
                                                   np.std(aa)))
        text += ("Kappa: {:.02f} +- {:.02f}\n".format(np.mean(kappas),
                                                      np.std(kappas)))
    else:
        text += "OA : {:.02f}%\n".format(Accuracy)
        text += "AA: {:.02f}%\n".format(aa)
        text += "Kappa: {:.02f}\n".format(kappa)
    print(text)


def predVisIN(index_true, label_true):
    image = np.zeros((145, 145, 3))

    # 类别与颜色的对应关系字典
    color_map = {
        16: [255, 255, 255],
        0: [219, 94, 86],
        1: [219, 144, 86],
        2: [219, 194, 86],
        3: [194, 219, 86],
        4: [145, 219, 86],
        5: [95, 219, 86],
        6: [86, 219, 127],
        7: [86, 219, 177],
        8: [86, 211, 219],
        9: [86, 161, 219],
        10: [86, 111, 219],
        11: [111, 86, 219],
        12: [160, 86, 219],
        13: [210, 86, 219],
        14: [219, 86, 178],
        15: [219, 86, 128]
    }
    # 遍历每个像素点
    for index, label in zip(index_true, label_true):
        # 获取对应的颜色
        color = color_map[label] if label in color_map else [0, 0, 0]
        # 将颜色赋值给图像的对应位置
        image[index[0], index[1]] = np.array(color) / 255.

    return image


def predVisUP(index_true, label_true):
    image = np.zeros((610, 340, 3))

    # 类别与颜色的对应关系字典
    color_map = {
        9: [255, 255, 255],
        0: [219, 94, 86],
        1: [219, 183, 86],
        2: [167, 219, 86],
        3: [86, 219, 94],
        4: [86, 219, 183],
        5: [86, 167, 219],
        6: [94, 86, 219],
        7: [183, 86, 219],
        8: [219, 86, 167]
    }

    # 遍历每个像素点
    for index, label in zip(index_true, label_true):
        # 获取对应的颜色
        color = color_map[label] if label in color_map else [0, 0, 0]
        # 将颜色赋值给图像的对应位置
        image[index[0], index[1]] = np.array(color) / 255.

    return image

def predVisKSC(index_true, label_true):
    image = np.zeros((512, 614, 3))

    # 类别与颜色的对应关系字典
    color_map = {
        13: [0, 0, 0],
        0: [219, 94, 86],
        1: [219, 155, 86],
        2: [219, 217, 86],
        3: [160, 219, 86],
        4: [99, 219, 86],
        5: [86, 219, 135],
        6: [86, 219, 196],
        7: [86, 180, 219],
        8: [86, 119, 219],
        9: [115, 86, 219],
        10: [176, 86, 219],
        11: [219, 86, 201],
        12: [219, 86, 139]
    }

    # 遍历每个像素点
    for index, label in zip(index_true, label_true):
        # 获取对应的颜色
        color = color_map[label] if label in color_map else [0, 0, 0]
        # 将颜色赋值给图像的对应位置
        image[index[0], index[1]] = np.array(color) / 255.

    return image


def display_predicted_colors(index_true, label_true):
    image = np.zeros((145, 145, 3))

    # 类别与颜色的对应关系字典
    color_map = {
        16: [255, 255, 255],
        0: [219, 94, 86],
        1: [219, 144, 86],
        2: [219, 194, 86],
        3: [194, 219, 86],
        4: [145, 219, 86],
        5: [95, 219, 86],
        6: [86, 219, 127],
        7: [86, 219, 177],
        8: [86, 211, 219],
        9: [86, 161, 219],
        10: [86, 111, 219],
        11: [111, 86, 219],
        12: [160, 86, 219],
        13: [210, 86, 219],
        14: [219, 86, 178],
        15: [219, 86, 128]
    }
    # 遍历每个像素点
    for index, label in zip(index_true, label_true):
        # 获取对应的颜色
        color = color_map[label] if label in color_map else [0, 0, 0]
        # 将颜色赋值给图像的对应位置
        image[index[0], index[1]] = np.array(color) / 255.
    return image


def display_predictions(pred, vis, gt=None, caption=""):
    # if gt is None:
    #     vis.images([np.transpose(pred, (2, 0, 1))],
    #                 opts={'caption': caption})
    # else:
    #     vis.images([np.transpose(pred, (2, 0, 1)),
    #                 np.transpose(gt, (2, 0, 1))],
    #                 nrow=2,
    #                 opts={'caption': caption})

    res = np.transpose(pred, (0, 1, 2))
    plt.imshow(res)
    plt.axis('off') # 不显示坐标轴
    plt.title(caption)  # 添加标题作为图片说明
    # plt.show()  # 显示图片

    vis.images([np.transpose(pred, (2, 0, 1))],
               opts={'caption': caption})



