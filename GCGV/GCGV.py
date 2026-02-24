import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
import numpy as np
import time
import os
from model import Branch_one, Branch_two
from model.Branch_two import ViT, GCN
from model.util import Normalize, get_data, A_Process, Metrics, Result_Show, Train_Test_Data, Train, Valid, Metric, PCA_Process

from Graph import MSS, create_graph
import yaml
from read_Data import data_read, data_split


config = '/home/JiLiang/GCGV/config/config.yaml'
with open(config, 'r', encoding='utf-8') as fin:
    configs = yaml.load(fin, Loader=yaml.FullLoader)
dataset_name = configs["data_input"]["dataset_name"]
epoches = configs["network_config"]["max_epoch"]

parser = argparse.ArgumentParser("HSI")

if dataset_name == "IndianPines":
    parser.add_argument('--dataset', choices=['Indian', 'PaviaU', 'KSC'],
                         default='Indian', help='dataset to use')
elif dataset_name == "paviaU":
    parser.add_argument('--dataset', choices=['Indian', 'PaviaU', 'KSC'],
                        default='PaviaU', help='dataset to use')
elif dataset_name == "Salinas":
    parser.add_argument('--dataset', choices=['Indian', 'PaviaU', 'KSC'],
                        default='Salinas', help='dataset to use')
elif dataset_name == "pavia":
    parser.add_argument('--dataset', choices=['Indian', 'PaviaU', 'KSC'],
                        default='Pavia', help='dataset to use')
elif dataset_name == "KSC":
    parser.add_argument('--dataset', choices=['Indian', 'PaviaU', 'KSC'],
                        default='KSC', help='dataset to use')

parser.add_argument('--patches', type=int, default=9, help='number of patches')
parser.add_argument('--n_gcn', type=int, default=21, help='number of related pix')
parser.add_argument('--pca_band', type=int, default=70, help='pca_components')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')

parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=10, help='number of evaluation')

parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--path-config', type=str, default=r'./config/config.yaml')
parser.add_argument('-pc', '--print-config', action='store_true', default=False)
parser.add_argument('-pdi','--print-data-info', action='store_true', default=False)
args = parser.parse_args()


def load_data():
    if dataset_name == "IndianPines":
        data = data_read.IndianRaw().normal_cube
        data_gt = data_read.IndianRaw().truth

    elif dataset_name == "paviaU":
        data = data_read.PaviaURaw().normal_cube
        data_gt = data_read.PaviaURaw().truth

    elif dataset_name == "Salinas":
        data = data_read.SalinasRaw().normal_cube
        data_gt = data_read.SalinasRaw().truth

    elif dataset_name == "pavia":
        data = data_read.PaviaRaw().normal_cube
        data_gt = data_read.PaviaRaw().truth

    elif dataset_name == "KSC":
        data = data_read.KSCRaw().normal_cube
        data_gt = data_read.KSCRaw().truth

    return data, data_gt


data, data_gt = load_data()
class_num = np.max(data_gt)
height, width, bands = data.shape #145,145,200
gt_reshape = np.reshape(data_gt, [-1])

config = yaml.load(open(args.path_config, "r"), Loader=yaml.FullLoader)
dataset_name = config["data_input"]["dataset_name"]
samples_type = config["data_split"]["samples_type"]
train_num = config["data_split"]["train_num"]
train_num1 = config["data_split"]["train_num1"]
train_num2 = config["data_split"]["train_num2"]

val_num = config["data_split"]["val_num"]
train_ratio = config["data_split"]["train_ratio"]
val_ratio = config["data_split"]["val_ratio"]
superpixel_scale = config["data_split"]["superpixel_scale"]
max_epoch = config["network_config"]["max_epoch"]
learning_rate = config["network_config"]["learning_rate"]
weight_decay = config["network_config"]["weight_decay"]
lb_smooth = config["network_config"]["lb_smooth"]

alfa = config["weight parameter"]["alfa"]
beta = config["weight parameter"]["beta"]
branch1 = config["weight parameter"]["branch1"]
branch2 = config["weight parameter"]["branch2"]

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
if args.print_config:
    print(config)

if dataset_name == "IndianPines":
    superpixel_scale = 100

elif dataset_name == "paviaU":
    superpixel_scale = 150

elif dataset_name == "Salinas":
    superpixel_scale = 100

elif dataset_name == "pavia":
    superpixel_scale = 500

elif dataset_name == "KSC":
    superpixel_scale = 200

train_index, val_index, test_index = data_split.Data_Split(gt_reshape,
                class_num, train_ratio, val_ratio, train_num1, train_num2, val_num, samples_type)


train_samples_gt, test_samples_gt, val_samples_gt = create_graph.label(gt_reshape,
                                                 train_index, val_index, test_index)

train_label_mask, test_label_mask, val_label_mask = create_graph.label_mask(train_samples_gt,
                                            test_samples_gt, val_samples_gt, data_gt, class_num)


train_gt = np.reshape(train_samples_gt, [height, width])
test_gt = np.reshape(test_samples_gt, [height, width])
val_gt = np.reshape(val_samples_gt, [height, width])

if args.print_data_info:
    data_read.Data_Index(train_gt, val_gt, test_gt)

train_gt_onehot = create_graph.label_to_one_hot(train_gt, class_num)
test_gt_onehot = create_graph.label_to_one_hot(test_gt, class_num)
val_gt_onehot = create_graph.label_to_one_hot(val_gt, class_num)

ls = MSS.SLIC_LDA(data, train_gt, class_num-1)

Q, S, A, Seg = ls.MSS_Process(scale=superpixel_scale)

Q = torch.from_numpy(Q).to(args.device)
A = torch.from_numpy(A).to(args.device)

train_samples_gt = torch.from_numpy(train_samples_gt.astype(np.float32)).to(args.device)
test_samples_gt = torch.from_numpy(test_samples_gt.astype(np.float32)).to(args.device)
val_samples_gt = torch.from_numpy(val_samples_gt.astype(np.float32)).to(args.device)

train_gt_onehot = torch.from_numpy(train_gt_onehot.astype(np.float32)).to(args.device)
test_gt_onehot = torch.from_numpy(test_gt_onehot.astype(np.float32)).to(args.device)
val_gt_onehot = torch.from_numpy(val_gt_onehot.astype(np.float32)).to(args.device)

train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(args.device)
test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(args.device)
val_label_mask = torch.from_numpy(val_label_mask.astype(np.float32)).to(args.device)

net_input = np.array(data, np.float32)
branch1_input = torch.from_numpy(net_input.astype(np.float32)).to(args.device)

# Branch1 model
branch1_net = Branch_one.Branch1(height, width, bands, class_num, Q, A).to(args.device)

optimizer3 = torch.optim.Adam(branch1_net.parameters(),lr=learning_rate, weight_decay=weight_decay)


np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False

input, num_classes, total_pos_train, total_pos_test, total_pos_true, y_train, y_test, y_true = get_data(args.dataset)

index_train = total_pos_train
index_test = total_pos_test
index_true = total_pos_true

label_train = y_train
label_test = y_test
label_true = y_true

pos_train = []
pos_test = []

input = PCA_Process(input, numComponents=args.pca_band)
print(input.shape)

input_normalize = Normalize(input)
height, width, band = input_normalize.shape

print("dataset:" + args.dataset)
print("height={0},width={1},band={2}".format(height, width, band))


x_train_band, x_test_band, x_true_band, corner_train, corner_test, corner_true, indexs_train, indexs_test, indexs_ture = Train_Test_Data(
    input_normalize, band, total_pos_train, total_pos_test, total_pos_true, patch=args.patches, w=height, h=width,
    n_gcn=args.n_gcn)

input2 = torch.from_numpy(input_normalize).type(torch.FloatTensor)

A_train = A_Process(x_train_band, input2, corner=corner_train,patches=args.patches , l=3,sigma=10)
x_train = torch.from_numpy(x_train_band).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.LongTensor)
total_pos_train = torch.from_numpy(total_pos_train).type(torch.LongTensor)
Label_train = Data.TensorDataset(A_train, x_train, y_train, total_pos_train)

A_test = A_Process(x_test_band, input2, corner=corner_test, patches=args.patches ,l=3, sigma=10)
x_test = torch.from_numpy(x_test_band).type(torch.FloatTensor)
y_test = torch.from_numpy(y_test).type(torch.LongTensor)
total_pos_test = torch.from_numpy(total_pos_test).type(torch.LongTensor)
Label_test = Data.TensorDataset(A_test, x_test, y_test, total_pos_test)

x_true = torch.from_numpy(x_true_band).type(torch.FloatTensor)
y_true = torch.from_numpy(y_true).type(torch.LongTensor)
total_pos_true = torch.from_numpy(total_pos_true).type(torch.LongTensor)
Label_true = Data.TensorDataset(x_true, y_true, total_pos_true)

label_train_loader = Data.DataLoader(Label_train, batch_size=args.batch_size, shuffle=True)
label_test_loader = Data.DataLoader(Label_test, batch_size=args.batch_size, shuffle=True)
label_true_loader = Data.DataLoader(Label_true, batch_size=100, shuffle=False)

results = []

best_OA2 = 0.0
best_AA_mean2 = 0.0
best_Kappa2 = 0.0

# Branch2 model

GCN_net = GCN(height, width, band, num_classes)
GCN_net = GCN_net.cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer1 = torch.optim.Adam(GCN_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=epoches // 10, gamma=args.gamma)


branch2_net = ViT(
    n_gcn=args.n_gcn,
    num_patches=64,
    num_classes=num_classes,
    dim=64,
    depth=5,
    heads=4,
    mlp_dim=8,
    dropout=0.1,
    emb_dropout=0.1,
)
branch2_net = branch2_net.cuda()

optimizer2 = torch.optim.Adam(branch2_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=epoches // 10, gamma=args.gamma)

print("start training")
tic = time.time()
for epoch in range(epoches):
    scheduler1.step()
    scheduler2.step()

    # train model
    GCN_net.train()
    branch2_net.train()
    branch1_net.train()

    train_acc, train_obj, tar_t, pre_t, pos_t, pos_train = Train(GCN_net, branch2_net, branch1_net, branch1_input, label_train_loader, criterion, optimizer1,
                                                optimizer2, optimizer3, indexs_train, branch1, branch2)

    OA1, AA_mean1, Kappa1, AA1 = Metric(tar_t, pre_t)
    print("Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}"
          .format(epoch + 1, train_obj, train_acc))

    if (epoch % args.test_freq == 0) | (epoch == epoches - 1) and epoch >= epoches*0.6:

        GCN_net.eval()
        branch2_net.eval()
        branch1_net.eval()
        tar_v, pre_v, pos_test = Valid(GCN_net, branch2_net, branch1_net, branch1_input, label_test_loader, criterion, indexs_test, branch1, branch2)
        OA2, AA_mean2, Kappa2, AA2 = Metric(tar_v, pre_v)

        if OA2 >= best_OA2 and AA_mean2 >= best_AA_mean2 and Kappa2 >= best_Kappa2:
            best_OA2 = OA2
            best_AA_mean2 = AA_mean2
            best_Kappa2 = Kappa2
            run_results = Metrics(best_OA2, best_AA_mean2, best_Kappa2,AA2)

Result_Show(run_results, agregated=False)
results.append(run_results)
toc = time.time()

#****************************************************************************************************#
#可视化相关操作
import visdom
from model.util import display_predictions, predVisIN, display_predicted_colors, predVisUP, predVisKSC
import matplotlib.pyplot as plt

all_pred = np.concatenate((pre_t, pre_v))
all_index = np.concatenate((pos_train, pos_test))


if dataset_name == "IndianPines":
    viz = visdom.Visdom(env=str("IP") + ' ' + str("GCGV"))
    y_pred = predVisIN(all_index, all_pred)

elif dataset_name == "paviaU":
    viz = visdom.Visdom(env=str("UP") + ' ' + str("GCGV"))
    y_pred = predVisUP(all_index, all_pred)

elif dataset_name == "KSC":
    viz = visdom.Visdom(env=str("KSC") + ' ' + str("GCGV"))
    y_pred = predVisKSC(all_index, all_pred)

#真实地物类别图像
# y_pred = display_predicted_colors(index_true, label_true)

display_predictions(y_pred, viz, gt="", caption="")
# print(y_pred.shape) #[145,145,3]

plt.imshow(y_pred)
plt.axis('off')
#plt.savefig('IP.png', bbox_inches='tight', pad_inches=0)
#plt.show()

