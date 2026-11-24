from einops import rearrange
import numpy as np
import matplotlib.pyplot as plt
from operator import truediv
import scipy.io as sio
import torch
import torch.utils.data as Data
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def _th_mean_std_normalize(image, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)):
    """ this version faster than torchvision.transforms.functional.normalize


    Args:
        image: 3-D or 4-D array of shape [batch (optional) , height, width, channel]
        mean:  a list or tuple or ndarray
        std: a list or tuple or ndarray

    Returns:

    """
    shape = [1] * image.dim()
    shape[-1] = -1
    mean = torch.tensor(mean, requires_grad=False).reshape(*shape)
    std = torch.tensor(std, requires_grad=False).reshape(*shape)

    return image.sub(mean).div(std)


def _np_mean_std_normalize(image, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)):
    """

    Args:
        image: 3-D array of shape [height, width, channel]
        mean:  a list or tuple or ndarray
        std: a list or tuple or ndarray

    Returns:

    """
    if not isinstance(mean, np.ndarray):
        mean = np.array(mean, np.float32)
    if not isinstance(std, np.ndarray):
        std = np.array(std, np.float32)
    shape = [1] * image.ndim
    shape[-1] = -1
    return (image - mean.reshape(shape)) / std.reshape(shape)


def mean_std_normalize(image, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)):
    """

    Args:
        image: 3-D array of shape [height, width, channel]
        mean:  a list or tuple
        std: a list or tuple

    Returns:

    """
    if isinstance(image, np.ndarray):
        return _np_mean_std_normalize(image, mean, std)
    elif isinstance(image, torch.Tensor):
        return _th_mean_std_normalize(image, mean, std)
    else:
        raise ValueError('The type {} is not support'.format(type(image)))


# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX


def divisible_pad(image_list, size_divisor=128, to_tensor=True):
    """

    Args:
        image_list: a list of images with shape [channel, height, width]
        size_divisor: int
        to_tensor: whether to convert to tensor
    Returns:
        blob: 4-D ndarray of shape [batch, channel, divisible_max_height, divisible_max_height]
    """
    max_shape = np.array([im.shape for im in image_list]).max(axis=0)

    max_shape[1] = int(np.ceil(max_shape[1] / size_divisor) * size_divisor)
    max_shape[2] = int(np.ceil(max_shape[2] / size_divisor) * size_divisor)

    if to_tensor:
        storage = torch.FloatStorage._new_shared(len(image_list) * np.prod(max_shape))
        out = torch.Tensor(storage).view([len(image_list), max_shape[0], max_shape[1], max_shape[2]])
        out = out.zero_()
    else:
        out = np.zeros([len(image_list), max_shape[0], max_shape[1], max_shape[2]], np.float32)

    for i, resized_im in enumerate(image_list):
        out[i, :, 0:resized_im.shape[1], 0:resized_im.shape[2]] = torch.from_numpy(resized_im)

    return out


def load_dataset(PC=30, rate=None, applypca=False, flag=None, disjoin=True):
    if disjoin:
        if flag == "1":
            image_dir = './data/disjoin/Indian_pines.mat'
            train_dir = './data/disjoin/IP_training.mat'
            test_dir = './data/disjoin/IP_testing.mat'
        if flag == "2":
            image_dir = './data/disjoin/PaviaU.mat'
            train_dir = './data/disjoin/PU_training.mat'
            test_dir = './data/disjoin/PU_testing.mat'
        if flag == "3":
            image_dir = './data/disjoin/UH2013w.mat'
            train_dir = './data/disjoin/UH2013_TR.mat'
            test_dir = './data/disjoin/UH2013_TE.mat'

        im_type = ".mat"
        my_array = np.array([1, 1])

        data_dict1 = sio.loadmat(image_dir)  # need an r!
        for key in data_dict1.keys():
            if type(data_dict1[key]) == type(my_array):
                data_hsi = data_dict1[key]
                if image_dir.split("\\")[-1] == "xiongan.mat":
                    data_hsi = rearrange(data_hsi, 'c h w -> h w c')
        data_dict2 = sio.loadmat(train_dir)
        for key in data_dict2.keys():
            if type(data_dict2[key]) == type(my_array):
                TR_gt = data_dict2[key].astype(float)
        data_dict3 = sio.loadmat(test_dir)
        for key in data_dict3.keys():
            if type(data_dict3[key]) == type(my_array):
                TE_gt = data_dict3[key].astype(float)

    else:
        my_array = np.array([1, 1])
        if flag == "1":
            image_dir = './data/Salinas.mat'
            train_dir = './data/Salinas_gt.mat'
            rate = 0.01
            im_type = ".mat"

        data_dict1 = sio.loadmat(image_dir)  # need an r!
        for key in data_dict1.keys():
            if type(data_dict1[key]) == type(my_array):
                data_hsi = data_dict1[key]
                if image_dir.split("\\")[-1] == "xiongan.mat":
                    data_hsi = rearrange(data_hsi, 'c h w -> h w c')
        data_dict2 = sio.loadmat(train_dir)
        for key in data_dict2.keys():
            if type(data_dict2[key]) == type(my_array):
                lab = data_dict2[key]

        TR_gt = np.zeros([lab.shape[0], lab.shape[1]], dtype=float)
        TE_gt = np.zeros([lab.shape[0], lab.shape[1]], dtype=float)
        for i in range(25):
            idx, idy = np.where(lab == i + 1)
            ID = np.random.permutation(len(idx))
            idx = idx[ID]
            idy = idy[ID]
            if rate > 1:
                if rate > len(idx):
                    tr_x = idx[0:15]
                    tr_y = idy[0:15]
                    te_x = idx[15:-1]
                    te_y = idy[15:-1]
                else:
                    tr_x = idx[0:rate]
                    tr_y = idy[0:rate]
                    te_x = idx[rate:-1]
                    te_y = idy[rate:-1]
            else:
                tr_x = idx[0:int(len(idx) * rate) + 1]
                tr_y = idy[0:int(len(idx) * rate) + 1]
                te_x = idx[int(len(idx) * rate) + 1:-1]
                te_y = idy[int(len(idx) * rate) + 1:-1]
            for j in range(len(tr_x)):
                TR_gt[tr_x[j], tr_y[j]] = lab[tr_x[j], tr_y[j]]
            for j in range(len(te_x)):
                TE_gt[te_x[j], te_y[j]] = lab[te_x[j], te_y[j]]

    # mean - std
    # im_cmean = data_hsi.reshape((-1, data_hsi.shape[-1])).mean(axis=0)
    # im_cstd = data_hsi.reshape((-1, data_hsi.shape[-1])).std(axis=0)
    # data_hsi = mean_std_normalize(data_hsi, im_cmean, im_cstd)

    # 区间缩放，返回值为缩放到[0, 1]区间的数据
    image_x, image_y, BAND = data_hsi.shape
    data_hsi = data_hsi.reshape((image_x * image_y, BAND))
    data_hsi = MinMaxScaler().fit_transform(data_hsi)
    data_hsi = data_hsi.reshape((image_x, image_y, BAND))
    if applypca:
        data_hsi = applyPCA(data_hsi, numComponents=PC)

    TOTAL_SIZE = TR_gt.shape[0] * TR_gt.shape[1]
    TRAIN_SIZE = TR_gt[TR_gt > 0].size
    TEST_SIZE = TE_gt[TE_gt > 0].size
    GT = TR_gt + TE_gt
    VALIDATION_SPLIT = 0
    return data_hsi, GT, TOTAL_SIZE, TRAIN_SIZE, TEST_SIZE, VALIDATION_SPLIT, TR_gt, TE_gt


def save_cmap(img, cmap, fname):
    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(img, cmap=cmap)
    plt.savefig(fname, dpi=height)
    plt.close()


def sampling(proportion, ground_truth, TE_gt, TR_gt, map_all=True):
    train = {}
    test = {}
    val = {}
    labels_loc = {}
    if proportion != 1:
        m = np.max(TE_gt)
        for i in range(int(m)):
            TR_indexes = [j for j, x in enumerate(TR_gt.ravel().tolist()) if x == i + 1]
            TE_indexes = [j for j, x in enumerate(TE_gt.ravel().tolist()) if x == i + 1]
            np.random.shuffle(TR_indexes)
            np.random.shuffle(TE_indexes)
            train[i] = TR_indexes[:]
            test[i] = TE_indexes[:]
            val[i] = TE_indexes[:]# * 0.1
        train_indexes = []
        test_indexes = []
        val_indexes = []
        for i in range(int(m)):
            train_indexes += train[i]
            test_indexes += test[i]
            val_indexes += val[i]

    else:
        if map_all:
            m = np.max(TE_gt)
            for i in range(int(m + 1)):
                TE_indexes = [j for j, x in enumerate(TE_gt.ravel().tolist()) if x == i]
                np.random.shuffle(TE_indexes)
                train[i] = TE_indexes[:1]
                test[i] = TE_indexes[:]
                val[i] = TE_indexes[:] #-int(len(TE_indexes))* 0.1
            train_indexes = []
            test_indexes = []
            val_indexes = []
            for i in range(int(m + 1)):
                train_indexes += train[i]
                test_indexes += test[i]
                val_indexes += val[i]
        else:
            m = np.max(TE_gt)
            for i in range(int(m)):
                TE_indexes = [j for j, x in enumerate(TE_gt.ravel().tolist()) if x == i + 1]
                np.random.shuffle(TE_indexes)
                train[i] = TE_indexes[:1]
                test[i] = TE_indexes[:]
                val[i] = TE_indexes[:]#-int(len(TE_indexes)) * 0.1
            train_indexes = []
            test_indexes = []
            val_indexes = []
            for i in range(int(m)):
                train_indexes += train[i]
                test_indexes += test[i]
                val_indexes += val[i]

    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    np.random.shuffle(val_indexes)

    trgt_mask_flatten = TR_gt.ravel()
    tegt_mask_flatten = TE_gt.ravel()
    train_indicator = np.zeros_like(trgt_mask_flatten)
    test_indicator = np.zeros_like(tegt_mask_flatten)
    val_indicator = np.zeros_like(tegt_mask_flatten)
    train_indicator[train_indexes] = 1
    test_indicator[test_indexes] = 1
    val_indicator[val_indexes] = 1

    train_indicator = train_indicator.reshape(ground_truth.shape)
    test_indicator = test_indicator.reshape(ground_truth.shape)
    val_indicator = val_indicator.reshape(ground_truth.shape)
    return train_indicator, test_indicator, val_indicator


def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi, ground_truth.shape[0] * 2.0 / dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0


def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 2:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 3:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 4:
            y[index] = np.array([0, 255, 255]) / 255.
        if item == 5:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 6:
            y[index] = np.array([192, 192, 192]) / 255.
        if item == 7:
            y[index] = np.array([128, 128, 128]) / 255.
        if item == 8:
            y[index] = np.array([128, 0, 0]) / 255.
        if item == 9:
            y[index] = np.array([128, 128, 0]) / 255.
        if item == 10:
            y[index] = np.array([0, 128, 0]) / 255.
        if item == 11:
            y[index] = np.array([128, 0, 128]) / 255.
        if item == 12:
            y[index] = np.array([0, 128, 128]) / 255.
        if item == 13:
            y[index] = np.array([0, 0, 128]) / 255.
        if item == 14:
            y[index] = np.array([255, 165, 0]) / 255.
        if item == 15:
            y[index] = np.array([255, 215, 0]) / 255.
        if item == 16:
            y[index] = np.array([105, 45, 194]) / 255.
        if item == 17:
            y[index] = np.array([215, 255, 0]) / 255.
        if item == 18:
            y[index] = np.array([0, 255, 215]) / 255.
        if item == 19:
            y[index] = np.array([192, 0, 0]) / 255.
        if item == 20:
            y[index] = np.array([192, 192, 0]) / 255.
        if item == 21:
            y[index] = np.array([192, 0, 192]) / 255.
        if item == -1:
            y[index] = np.array([0, 0, 0]) / 255.
    return y


def generate_iter(train_indices, test_indices, val_indices, total_indices,
                  whole_data, gt, TR_gt, TE_gt, flag):
    gt_all = gt - 1
    y_train = TR_gt - 1
    y_test = TE_gt - 1

    x_train = whole_data
    x_test_all = whole_data

    x_val = x_test_all
    y_val = y_test

    x_test = x_test_all
    y_test = y_test

    x1_tensor_train = torch.from_numpy(x_train).type(torch.FloatTensor).unsqueeze(0)
    y1_tensor_train = torch.from_numpy(y_train).type(torch.FloatTensor).unsqueeze(0)
    y1_tensor_train_indices = torch.from_numpy(train_indices).type(torch.FloatTensor).unsqueeze(0)
    torch_dataset_train = Data.TensorDataset(x1_tensor_train, y1_tensor_train, y1_tensor_train_indices)

    x1_tensor_valida = torch.from_numpy(x_val).type(torch.FloatTensor).unsqueeze(0)
    y1_tensor_valida = torch.from_numpy(y_val).type(torch.FloatTensor).unsqueeze(0)
    y1_tensor_val_indices = torch.from_numpy(val_indices).type(torch.FloatTensor).unsqueeze(0)
    torch_dataset_valida = Data.TensorDataset(x1_tensor_valida, y1_tensor_valida, y1_tensor_val_indices)

    x1_tensor_test = torch.from_numpy(x_test).type(torch.FloatTensor).unsqueeze(0)
    y1_tensor_test = torch.from_numpy(y_test).type(torch.FloatTensor).unsqueeze(0)
    y1_tensor_test_indices = torch.from_numpy(test_indices).type(torch.FloatTensor).unsqueeze(0)
    torch_dataset_test = Data.TensorDataset(x1_tensor_test, y1_tensor_test, y1_tensor_test_indices)

    train_iter = Data.DataLoader(
        dataset=torch_dataset_train,  # torch TensorDataset format
        batch_size=1,  # mini batch size
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
        num_workers=1,  # 多线程来读数据
        pin_memory=True,
    )
    valiada_iter = Data.DataLoader(
        dataset=torch_dataset_valida,  # torch TensorDataset format
        batch_size=1,  # mini batch size
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
        num_workers=1,  # 多线程来读数据
        pin_memory=True,
    )
    test_iter = Data.DataLoader(
        dataset=torch_dataset_test,  # torch TensorDataset format
        batch_size=1,  # mini batch size
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
        num_workers=1,  # 多线程来读数据
        pin_memory=True,
    )
    return train_iter, valiada_iter, test_iter


def generate_png(net, gt, device, test_iter, total_indices, flag, map_all):
    for X, y, w in test_iter:
        X = X.to(device)
        net.eval()  # 评估模式, 这会关闭dropout
        y_pred,x_s = net(X)
        y_pred = y_pred.argmax(dim=1).cpu()
    x = np.ravel(y_pred)
    # 循环结束
    yy = x.reshape(gt.shape[0], gt.shape[1])
    from PIL import Image
    yy = yy.astype(np.uint8)
    im = Image.fromarray(yy)

    y_list = list_to_colormap(x)
    y_re = np.reshape(y_list, (gt.shape[0], gt.shape[1], 3))

    import datetime
    day = datetime.datetime.now()
    day_str = day.strftime('%m_%d_%H_%M_%S_')
    # 存tif
    im.save('./classification_maps/' + day_str + '_' + flag + '.tif')
    # 存png
    classification_map(y_re, gt, 300, './classification_maps/' + day_str + '_' + flag + '.png')
    print('------Get classification maps successful-------')

    # ## 生成中间参数
    # for X, y, w in test_iter:
    #     X = X.to(device)
    #     net.eval()  # 评估模式, 这会关闭dropout
    #     y_pred,xout,x_s = net(X, test=True)
    
    # y_pred = y_pred.cpu().detach().numpy()
    # xout = xout.cpu().detach().numpy()
    # x_s = x_s.cpu().detach().numpy()

    # sio.savemat('./classification_maps/' + day_str + '_xout_' + flag + '.mat', {'xout':y_pred})
    # sio.savemat('./classification_maps/' + day_str + '_x_s_' + flag + '.mat', {'x_s':x_s})
    # print('------Get intermediate parameters successful-------')
    
