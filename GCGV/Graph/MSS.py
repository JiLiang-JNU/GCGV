import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from skimage.segmentation import slic, mark_boundaries
from sklearn import preprocessing
import yaml

# 想保存不同数据集上的超像素图片
config = '/home/JiLiang/GCGV/config/config.yaml'
with open(config, 'r', encoding='utf-8') as fin:
    configs = yaml.load(fin, Loader=yaml.FullLoader)

name = configs['data_input']['dataset_name']


def Label_Seg(labels):
    labels = np.array(labels, np.int64)
    H, W = labels.shape
    ls = list(set(np.reshape(labels, [-1]).tolist()))

    dic = {}
    for i in range(len(ls)):
        dic[ls[i]] = i

    new_labels = labels
    for i in range(H):
        for j in range(W):
            new_labels[i, j] = dic[new_labels[i, j]]
    return new_labels


class SLIC(object):
    def __init__(self, HSI, labels, n_segments=1000, compactness=20,
                 sigma=0, min_size_factor=0.3, max_size_factor=2):

        self.n_segments = n_segments
        self.compactness = compactness
        self.min_size_factor = min_size_factor
        self.max_size_factor = max_size_factor
        self.sigma = sigma
        height, width, bands = HSI.shape
        data = np.reshape(HSI, [height * width, bands])
        minMax = preprocessing.StandardScaler()
        data = minMax.fit_transform(data)
        self.data = np.reshape(data, [height, width, bands])
        self.labels = labels

    def Q_S_Segments(self):
        img = self.data
        (h, w, d) = img.shape

        segments = slic(img, n_segments=self.n_segments, compactness=self.compactness,
                        convert2lab=False, sigma=self.sigma, enforce_connectivity=True,
                        min_size_factor=self.min_size_factor, max_size_factor=self.max_size_factor, slic_zero=False,
                        start_label=0)

        if segments.max() + 1 != len(list(set(np.reshape(segments, [-1]).tolist()))):
            segments = Label_Seg(segments)
        self.segments = segments
        superpixel_count = segments.max() + 1
        self.superpixel_count = superpixel_count
        # print("superpixel_count", superpixel_count)

        ##################################### 显示超像素图片 ######################################
        out = mark_boundaries(img[:, :, [0, 1, 2]], segments)
        # plt.figure()
        # plt.imshow(out)
        # plt.savefig('./createGraph/result-'+name+'.png')
        # plt.show()
        ##################################### 显示超像素图片 ######################################

        segments = np.reshape(segments, [-1])
        S = np.zeros([superpixel_count, d], dtype=np.float32)
        Q = np.zeros([w * h, superpixel_count], dtype=np.float32)
        x = np.reshape(img, [-1, d])
        for i in range(superpixel_count):
            idx = np.where(segments == i)[0]
            count = len(idx)
            pixels = x[idx]
            superpixel = np.sum(pixels, 0) / count
            S[i] = superpixel
            Q[idx, i] = 1

        self.S = S
        self.Q = Q

        return Q, S, self.segments

    def A_Seg(self, sigma: float):
        
        A = np.zeros([self.superpixel_count, self.superpixel_count], dtype=np.float32)
        # print("A.shape", A.shape)
        (h, w) = self.segments.shape
        for i in range(h - 2):
            for j in range(w - 2):
                sub = self.segments[i:i + 2, j:j + 2]
                sub_max = np.max(sub).astype(np.int32)
                sub_min = np.min(sub).astype(np.int32)
                if sub_max != sub_min:
                    idx1 = sub_max
                    idx2 = sub_min
                    if A[idx1, idx2] != 0:
                        continue

                    pix1 = self.S[idx1]
                    pix2 = self.S[idx2]
                    diss = np.exp(-np.sum(np.square(pix1 - pix2)) / sigma ** 2)
                    A[idx1, idx2] = A[idx2, idx1] = diss

        return A


class SLIC_LDA(object):
    def __init__(self, data, labels, n_component):
        self.data = data
        self.init_labels = labels
        self.curr_data = data
        self.n_component = n_component
        self.height, self.width, self.bands = data.shape
        self.x_flatt = np.reshape(data, [self.width * self.height, self.bands])
        self.y_flatt = np.reshape(labels, [self.height * self.width])
        self.labes = labels
        # self.Q = np.zeros()
        # self.S = np.zeros()
        # self.A = np.zeros()
        # self.Segments = np.zeros()

    def LDA(self, curr_labels):
        curr_labels = np.reshape(curr_labels, [-1])
        idx = np.where(curr_labels != 0)[0]
        x = self.x_flatt[idx]
        y = curr_labels[idx]
        lda = LinearDiscriminantAnalysis(n_components=self.n_component)  # n_components = self.n_component
        lda.fit(x, y - 1)
        X_new = lda.transform(self.x_flatt)

        return np.reshape(X_new, [self.height, self.width, -1])

    def Split(self, img, num_rows, num_cols):

        height, width, channels = img.shape
        # assert height % num_rows == 0, "Height is not divisible by the number of rows."
        # assert width % num_cols == 0, "Width is not divisible by the number of columns."

        row_size = height // num_rows
        col_size = width // num_cols

        sub_images = []
        for i in range(num_rows):
            for j in range(num_cols):
                sub_img = img[i * row_size: (i + 1) * row_size, j * col_size: (j + 1) * col_size, :]
                sub_images.append(sub_img)

        return sub_images

    def SLIC_(self, img, scale=25):

        # n_segments_init = self.height*self.width/scale
        n_segments_init = img.shape[0] * img.shape[1] / scale

        # print("n_segments_init",n_segments_init)                      # 145*145/100 = 210.25

        myslic = SLIC(img, n_segments=n_segments_init, labels=self.labes, compactness=1, sigma=1, min_size_factor=0.1,
                      max_size_factor=2)
        Q, S, Segments = myslic.Q_S_Segments()
        A = myslic.A_Seg(sigma=10)

        self.Q = Q
        self.S = S
        self.A = A
        self.Segments = Segments
        return Q, S, A, Segments

    def Multiscale_SLIC(self, img, scale=25):

        num_rows = 5
        num_cols = 5
        sub_images = self.Split(img, num_rows, num_cols)

        all_Q, all_S, all_A, all_Segments = [], [], [], []
        Q1 = self.Q
        S1 = self.S
        A1 = self.A
        Segments1 = self.Segments

        row = 0
        col = 0
        for i, sub_img in enumerate(sub_images):

            # print(sub_img.shape) #[29,29,15]
            Q2, S2, A2, Segments2 = self.SLIC_(sub_img, scale=100)

            # 计算子图在大矩阵中的位置
            row = min(i * A2.shape[0], A1.shape[0] - A2.shape[0])
            col = min(i * A2.shape[1], A1.shape[1] - A2.shape[1])
            # row = (i // num_rows) * (A1.shape[0] // A2.shape[0])
            # col = (i % num_cols) * (A1.shape[0] // A2.shape[0])

            # 将结果累加到大矩阵的相应位置
            # Q1[row:row + 9, col:col + 9] += Q2
            # S1[row:row + 9, col:col + 9] += S2
            # A1[row:row + A2.shape[0], col:col + A2.shape[1]] += A2

            A1[row:row + A2.shape[0], col:col + A2.shape[1]] = np.maximum(
                A2[:A2.shape[0], :A2.shape[1]], A1[row:row + A2.shape[0], col:col + A2.shape[1]])

            # Segments1[row:row + 9, col:col + 9] += Segments2

            # print(Q1.shape)
            # print(S1.shape)
            # print(A1.shape)
            # print(Segments1.shape)

            if row == A1.shape[0] - A2.shape[0] and col == A1.shape[1] - A2.shape[1]:
                break

        return Q1, S1, A1, Segments1

    def MSS_Process(self, scale):
        curr_labels = self.init_labels
        X = self.LDA(curr_labels)
        print(X.shape)  # LDA降维：[145,145,15]
        Q, S, A, Seg = self.SLIC_(X, scale=scale)
        Q, S, A, Seg = self.Multiscale_SLIC(X, scale=scale)

        # print(Q2.shape)
        # print(S2.shape)
        # print(A2.shape)
        # print(Seg2.shape)

        return Q, S, A, Seg