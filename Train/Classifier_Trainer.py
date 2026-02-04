# Designer:Pan YuDong
# Coder:God's hand
# Time:2022/7/4 20:40
import torch
import time
import numpy as np
# 1. 引入 sklearn 计算指标
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

def train_on_batch(num_epochs, train_iter, valid_iter, lr, criterion, net, device, wd=0, lr_jitter=False):
    trainer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer, T_max=num_epochs * len(train_iter), eta_min=5e-6)
    #新增初始化最佳精度
    best_acc = 0.0
    best_kappa = 0.0  # 初始化最佳 Kappa
    best_f1 = 0.0  # 初始化最佳 F1

    best_preds = []
    best_labels = []
    #
    for epoch in range(num_epochs):
        # training
        net.train()
        sum_loss = 0.0
        sum_acc = 0.0
        for (X, y) in train_iter:
            X = X.type(torch.FloatTensor)
            y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64)
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            loss = criterion(y_hat, y).sum()
            trainer.zero_grad()
            loss.backward()
            #加入
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
            #
            trainer.step()
            if lr_jitter:
                scheduler.step()
            sum_loss += loss.item() / y.shape[0]
            sum_acc += (y == y_hat.argmax(dim=-1)).float().mean()
        train_loss = sum_loss / len(train_iter)
        train_acc = sum_acc / len(train_iter)

        # test
        #修改 改为每个epoch都计算一遍，删除了if epoch == num_epochs - 1:
        #if epoch == num_epochs - 1:
        all_preds = []
        all_labels = []
        net.eval()
        #val_sum_acc = 0.0
        with torch.no_grad():
            for (X, y) in valid_iter:
                X = X.type(torch.FloatTensor)
                y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64)
                X = X.to(device)
                y = y.to(device)
                y_hat = net(X)

                preds = y_hat.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                #val_sum_acc += (y == y_hat.argmax(dim=-1)).float().mean()
        #current_val_acc = val_sum_acc / len(valid_iter)
        current_val_acc = accuracy_score(all_labels, all_preds)
        current_kappa = cohen_kappa_score(all_labels, all_preds)
        current_f1 = f1_score(all_labels, all_preds, average='macro')
        #新增：比较并且保存最佳模型
        if current_val_acc > best_acc:
            best_acc = current_val_acc
            best_kappa = current_kappa
            best_f1 = current_f1

            best_preds = all_preds
            best_labels = all_labels
            # 如果你想保存权重文件，把下面这行注释取消掉：
            # torch.save(net.state_dict(), 'best_model.pth')
            print(f"  [New Best] Epoch {epoch + 1}: Acc updated to {best_acc:.3f}")
            # --------------------------------

        print(
            f"epoch{epoch + 1}, train_loss={train_loss:.3f}, train_acc={train_acc:.3f}, val_acc={current_val_acc:.3f}")
    print(
        f'training finished at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} with best_valid_acc={best_acc:.3f}')
    torch.cuda.empty_cache()
    return best_acc, best_kappa, best_f1, best_preds, best_labels
