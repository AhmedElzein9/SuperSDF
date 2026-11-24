import time
from torch import optim
from sklearn import preprocessing
import datetime
from utils import *
import mambaLG
import mambaLG_IP
from utils.generate_pic import sampling, load_dataset, generate_png, generate_iter
import torch.nn as nn

def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)


class SelfLoss(nn.Module):
    def __init__(self, device, class_num, gamma=2, l1_num=5):
        super(SelfLoss, self).__init__()
        self.device = device
        self.band_num = class_num
        self.gamma = gamma
        self.l1_num = l1_num

    def forward(self, y_pre, x_s, y, w):
        # b,c,h,w #
        # losses = F.cross_entropy(y_pre, y, weight=None,
        #                          ignore_index=-1, reduction='none')
        #
        # v = losses.mul(w).sum() / w.sum()

        # # focal
        # celoss = nn.CrossEntropyLoss(weight=None, ignore_index=-1, reduction='none')(y_pre, y)
        # pt = torch.exp(-celoss)
        # focal_loss = (1 - pt) ** self.gamma * celoss
        # v = focal_loss.mul(w).sum() / w.sum()

        # focal+l1
        threshold = 1.5  # 定义较大值的阈值:1.5
        required_count = self.l1_num  # 需要的较大值数量:5
        large_value_count = torch.sum(x_s > threshold)  # 计算大于阈值的元素数量
        count_constraint = torch.relu(required_count - large_value_count)  # 计算缺少的数量

        l1loss = torch.norm(x_s, p=1) / x_s.shape[3]
        celoss = nn.CrossEntropyLoss(weight=None, ignore_index=-1, reduction='none')(y_pre, y)
        pt = torch.exp(-celoss)
        focal_loss = (1 - pt) ** self.gamma * celoss
        v = focal_loss.mul(w).sum() / w.sum() + 2 * l1loss + count_constraint

        return v.to(self.device)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  #

seeds = [1]
ensemble = 1
torch.manual_seed(1)
torch.cuda.manual_seed(1)
day = datetime.datetime.now()
day_str = day.strftime('%m_%d_%H_%M_%S')
np.random.seed(1)
flag = ["1"]  # disjoin:"1", "2", "3"; join:"1"
rate = [0.5]  #
spec = [6]
dis = True
Map_All = False
for FLAG in flag:
    for RATE in rate:
        for SPEC in spec:
            data_hsi, gt, TOTAL_SIZE, TRAIN_SIZE, TEST_SIZE, VALIDATION_SPLIT, TR_gt, TE_gt = load_dataset(flag=FLAG,
                                                                                                           disjoin=dis)
            print(data_hsi.shape)
            image_x, image_y, BAND = data_hsi.shape
            data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))
            data = preprocessing.scale(data)
            data_ = data.reshape(data_hsi.shape[0], data_hsi.shape[1], data_hsi.shape[2])
            whole_data = data_

            CLASSES_NUM = int(gt.max())
            TR_gt = TR_gt.reshape(np.prod(TR_gt.shape[:2]), )
            TE_gt = TE_gt.reshape(np.prod(TE_gt.shape[:2]), )

            print('The class numbers of the HSI data is:', CLASSES_NUM)

            print('-----Importing Setting Parameters-----')

            ITER = 1
            # number of training samples per class
            lr, num_epochs = 0.0005, 1000

            print('Train size: ', TRAIN_SIZE)
            print('Test size: ', TEST_SIZE)

            ALL_SIZE = data_hsi.shape[0] * data_hsi.shape[1]
            VAL_SIZE = int(TRAIN_SIZE * 0.1)

            print('all_size: ', ALL_SIZE)
            print('Validation size: ', VAL_SIZE)

            KAPPA = []
            OA = []
            AA = []
            TRAINING_TIME = []
            TESTING_TIME = []
            ELEMENT_ACC = np.zeros((ITER, CLASSES_NUM))

            train_indices, test_indices, val_indices = sampling(VALIDATION_SPLIT, gt, TE_gt, TR_gt)
            _, total_indices, _ = sampling(1, gt, TE_gt, TR_gt, map_all=Map_All)
            TR_gt = TR_gt.reshape(gt.shape)
            TE_gt = TE_gt.reshape(gt.shape)

            print('-----Selecting Small Pieces from the Original Cube Data-----')

            train_iter, valida_iter, test_iter = generate_iter(train_indices, test_indices, val_indices, total_indices,
                                                               whole_data, gt, TR_gt, TE_gt, 1)

            for index_iter in range(ITER):
                print('iter:', index_iter)
                if dis:
                    if FLAG == '1':  # IP
                        net = mambaLG_IP.mambaLG(num_classes=CLASSES_NUM, dim=64, band=BAND, device=device, spec_num=12,
                                                 spec_rate=RATE, spa_token=12)
                    if FLAG == '2':  # PU
                        net = mambaLG.mambaLG(num_classes=CLASSES_NUM, dim=64, band=BAND, device=device, spec_num=12,
                                              spec_rate=RATE, spa_token=12)
                    if FLAG == '3':  # UH
                        net = mambaLG.mambaLG(num_classes=CLASSES_NUM, dim=84, band=BAND, device=device, spec_num=12,
                                              spec_rate=RATE, spa_token=12)
                else:
                    net = mambaLG.mambaLG(num_classes=CLASSES_NUM, dim=64, band=BAND, device=device, spec_num=12,
                                          spec_rate=RATE, spa_token=12)

                loss = SelfLoss(device, CLASSES_NUM, l1_num=SPEC)
                model_structure(net)
                optimizer = optim.Adam(net.parameters(), lr=lr, amsgrad=False)  # , weight_decay=0.0001)
                time_1 = int(time.time())

                tic1 = time.perf_counter()
                train(net, train_iter, valida_iter, loss, optimizer, device, epochs=num_epochs, early_stopping=False)
                toc1 = time.perf_counter()

                pred_test_fdssc = []
                tic2 = time.perf_counter()
                with torch.no_grad():
                    for X, y, w in test_iter:
                        X = X.to(device)
                        net.eval()  # 评估模式, 这会关闭dropout
                        y_pred, x_s = net(X)
                        # print(net(X))
                        y_pred = y_pred.argmax(dim=1).cpu() + 1
                        y = y + 1

                        # w.unsqueeze_(dim=0)
                        w = w > 0
                        y = torch.masked_select(y.view(-1), w.view(-1))
                        y_pred = torch.masked_select(y_pred.view(-1), w.view(-1))

                        oa = metricss.th_overall_accuracy_score(y.view(-1), y_pred.view(-1))
                        aa, acc_per_class = metricss.th_average_accuracy_score(y.view(-1), y_pred.view(-1),
                                                                               CLASSES_NUM,
                                                                               return_accuracys=True)
                        kappa = metricss.th_cohen_kappa_score(y.view(-1), y_pred.view(-1),
                                                              CLASSES_NUM)
                        print("OA %.6f,aa %.6f,kappa %.6f" % (oa, aa, kappa))
                        print(acc_per_class)
                        # generate_png(net, gt, device, test_iter, total_indices, FLAG, map_all=Map_All)

                toc2 = time.perf_counter()

                torch.save(net.state_dict(), "./net/" + str(round(float(oa), 3)) + '.pt')
                KAPPA.append(kappa)
                OA.append(oa)
                AA.append(aa)
                TRAINING_TIME.append(toc1 - tic1)
                TESTING_TIME.append(toc2 - tic2)
                ELEMENT_ACC[index_iter, :] = acc_per_class

            print("-------- Training Finished-----------")
            record.record_output(OA, AA, KAPPA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME,
                                 'records/' + day_str + '_' + str(FLAG) + "_SPEC_" + str(
                                     SPEC) + "_RATE_" + str(RATE) + '.txt')
            # del test_iter, train_iter, valida_iter
            generate_png(net, gt, device, test_iter, total_indices, FLAG, map_all=Map_All)
