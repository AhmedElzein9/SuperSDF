import time
import torch


def evaluate_accuracy(data_iter, net, loss, device):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y, w in data_iter:
            test_l_sum, test_num = 0, 0
            # X = X.to(device)
            y = y.to(device)
            w = w.to(device)
            net.eval()  # 评估模式, 这会关闭dropout
            y_hat, x_s = net(X.to(device))
            l = loss(y_hat, x_s, y.long(), w)
            acc_sum += (y_hat.argmax(dim=1) == y.to(device)).float().mul_(w).sum().cpu().item()
            test_l_sum += l.cpu().item()
            test_num += 1
            # net.train()  # 改回训练模式
            n += w.sum()  # y.shape[0]

    return [acc_sum / n, test_l_sum / test_num]  # / test_num]


def train(net, train_iter, valida_iter, loss, optimizer, device, epochs=30, early_stopping=True,
          early_num=20):
    loss_list = [100]
    early_epoch = 0

    net = net.to(device)
    print("training on ", device)
    start = time.time()
    train_loss_list = []
    valida_loss_list = []
    train_acc_list = []
    valida_acc_list = []

    for epoch in range(epochs):
        net.train()
        train_acc_sum, n = 0.0, 0
        time_epoch = time.time()
        lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=1e-6, last_epoch=-1)
        # lr_adjust = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200], gamma=0.4)
        for X, y, w in train_iter:
            batch_count, train_l_sum = 0, 0
            # X = X.to(device)
            y = y.to(device)
            w = w.to(device)
            y_hat, x_s = net(X.to(device))
            # print('y_hat', y_hat)
            # print('y', y)
            l = loss(y_hat, x_s, y.long(), w)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).float().mul_(w).sum().cpu().item()
            n += w.sum()
            batch_count += 1

        lr_adjust.step(epoch)
        valida_acc, valida_loss = evaluate_accuracy(valida_iter, net, loss, device)
        loss_list.append(valida_loss)

        # 绘图部分
        train_loss_list.append(train_l_sum / batch_count)  # / batch_count)
        train_acc_list.append(train_acc_sum / n)
        valida_loss_list.append(valida_loss)
        valida_acc_list.append(valida_acc)

        print('epoch %d, train loss %.6f, train acc %.3f, valida loss %.6f, valida acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, valida_loss, valida_acc,
                 time.time() - time_epoch))
