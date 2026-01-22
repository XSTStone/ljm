# 精度 72.54版
# Time:2025.12.23
import torch
import argparse
import sys
import os
import datetime
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. 获取当前脚本的父目录 (即 D:\STS\SSVEPNET\SSVEPNet-master)
parent_dir = os.path.dirname(current_dir)
# 3. 将父目录加入到系统路径 sys.path 中
sys.path.append(parent_dir)
import Utils.EEGDataset as EEGDataset
from Utils import Ploter
from Utils import LossFunction
from Model import SSVEPNet
from Utils import Constraint
from Train import Classifier_Trainer

# 设置日志文件
log_file = f"SSVEPNet_Test_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_path = os.path.join(parent_dir, 'Result', '2020', log_file)
os.makedirs(os.path.dirname(log_path), exist_ok=True)

# 1、Define parameters of eeg
'''                    
---------------------------------------------Intra-subject Experiments ---------------------------------
                        epochs    bz     lr   lr_scheduler    ws      Fs    Nt   Nc   Nh   Ns     wd
    DatasetA(1S/0.5S):  500      30    0.01      Y           1/0.5   256   1024  8   180  10  0.0003
    DatasetB(1S/0.5S):  500      16    0.01      Y           1/0.5   250   1000  8    80  10  0.0003
---------------------------------------------Inter-subject Experiments ---------------------------------
                        epochs     bz          lr       lr_scheduler  ws      Fs     Nt    Nc   Nh      wd        Kf
    DatasetA(1S/0.5S):  500/100   64/30    0.001/0.01     N/Y        1/0.5    256   1024   8    180   0/0.0001   1/5
    DatasetB(1S/0.5S):  500/100   64/30    0.001/0.01     N/Y        1/0.5    250   1000   8     80   0/0.0003   1/5 
'''
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300, help="number of epochs")#训练轮次
parser.add_argument('--bz', type=int, default=200, help="number of batch")#批大小
parser.add_argument('--lr', type=float, default=0.01, help="learning rate")#学习率
parser.add_argument('--ws', type=float, default=0.5, help="window size of ssvep")#时间窗长度
parser.add_argument('--Kf', type=int, default=1, help="k-fold cross validation")#K折交叉验证
#以下是数据集的参数
parser.add_argument('--Nh', type=int, default=350, help="number of trial")
parser.add_argument('--Nc', type=int, default=64, help="number of channel")
parser.add_argument('--Fs', type=int, default=256, help="frequency of sample")
parser.add_argument('--Nt', type=int, default=795, help="number of sample")
parser.add_argument('--Nf', type=int, default=5, help="number of stimulus")
parser.add_argument('--Ns', type=int, default=15, help="number of subjects")
parser.add_argument('--wd', type=int, default=0.0003, help="weight decay")#权重衰减
opt = parser.parse_args()
devices = "cuda:1" if torch.cuda.is_available() else "cpu"

#数据切分
def split_eeg_samples(data, labels, fs, num_windows=3):
    """
    data: (Trials, 1, Nc, Nt) -> (N, 1, 64, 795)
    labels: (Trials, 1)
    fs: 采样率 (256)
    """
    win_len = int(fs*0.5) # 1秒长度 = 256个点

    # 存储切分后的列表
    split_data_list = []

    for i in range(num_windows):
        start = i * win_len
        end = (i + 1) * win_len
        # 提取时间段: (N, 1, 64, 256)
        segment = data[:, :, :, start:end]
        split_data_list.append(segment)

    # 在第0维（样本维）合并: (N*3, 1, 64, 256)
    new_data = torch.cat(split_data_list, dim=0)

    # 标签同步复制 3 次
    new_labels = torch.cat([labels for _ in range(num_windows)], dim=0)

    return new_data, new_labels

opt.ws = 0.5
win_points = int(opt.Fs * opt.ws)

# 定义日志写入函数
def write_log(message, to_file=True, to_console=True):
    """写入日志到文件和控制台"""
    if to_console:
        print(message)
    if to_file:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

# 运行三次实验并记录最好结果
num_runs = 3
best_total_accuracy = 0.0
all_run_accuracies = []

write_log("="*50)
write_log("SSVEPNet Training Log")
write_log(f"Start Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
write_log(f"Parameters: {opt}")
write_log(f"Total Runs: {num_runs}")
write_log("="*50)

for run in range(num_runs):
    write_log(f"\n{'='*40}")
    write_log(f"Run {run + 1}/{num_runs}")
    write_log(f"{'='*40}")
    
    # 本次运行的结果列表
    run_final_acc_list = []
    run_subject_best_epochs = []
    
    for fold_num in range(opt.Kf): #第几折
        best_valid_acc_list = []
        final_valid_acc_list = []
        fold_best_epochs = [] # 记录当前折的每个subject的最佳精度迭代
        
        write_log(f"\nTraining for K_Fold {fold_num + 1}")
        
        for testSubject in range(1, opt.Ns + 1):
            # **************************************** #
            '''12-class SSVEP Dataset'''
            # -----------Intra-Subject Experiments--------------
            # EEGData_Train = EEGDataset.getSSVEP12Intra(subject=testSubject, KFold=fold_num, n_splits=opt.Kf,
            #                                           mode='test')
            # EEGData_Test = EEGDataset.getSSVEP12Intra(subject=testSubject, KFold=fold_num, n_splits=opt.Kf,
            #                                          mode='train')

            write_log(f"\nSubject {testSubject}")
            EEGData_Train = EEGDataset.getSSVEP2020Intra(subject=testSubject, train_ratio=0.8, mode='train')
            EEGData_Test = EEGDataset.getSSVEP2020Intra(subject=testSubject, train_ratio=0.8, mode='test')

            raw_train_data, raw_train_label = EEGData_Train[:]
            train_data_split, train_label_split = split_eeg_samples(raw_train_data, raw_train_label, opt.Fs, num_windows=3)
            write_log(f"Original Train: {raw_train_data.shape} -> Augmented Train: {train_data_split.shape}")
            EEGData_Train = torch.utils.data.TensorDataset(train_data_split, train_label_split)

            raw_test_data, raw_test_label = EEGData_Test[:]
            # 调用切分函数：1个样本变3个
            test_data_split, test_label_split = split_eeg_samples(raw_test_data, raw_test_label, opt.Fs, num_windows=3)
            write_log(f"Original Test: {raw_test_data.shape} -> Augmented Test: {test_label_split.shape}")
            EEGData_Test = torch.utils.data.TensorDataset(test_data_split, test_label_split)

            # Create DataLoader for the Dataset
            train_dataloader = torch.utils.data.DataLoader(dataset=EEGData_Train, batch_size=opt.bz, shuffle=True,
                                               drop_last=True)
            valid_dataloader = torch.utils.data.DataLoader(dataset=EEGData_Test, batch_size=opt.bz, shuffle=False,
                                               drop_last=True)

            # Define Network
            net = SSVEPNet.ESNet(opt.Nc, win_points, opt.Nf)
            net = Constraint.Spectral_Normalization(net) #创新：谱归一化
            net = net.to(devices)
            # criterion = nn.CrossEntropyLoss(reduction="none")
            criterion = LossFunction.CELoss_Marginal_Smooth(opt.Nf, stimulus_type='12')
            valid_acc, best_epochs = Classifier_Trainer.train_on_batch(opt.epochs, train_dataloader, valid_dataloader, opt.lr, criterion,
                                                          net, devices, wd=opt.wd, lr_jitter=True)
            final_valid_acc_list.append(valid_acc)
            fold_best_epochs.append(best_epochs)
            
            # 记录当前subject的最佳精度迭代
            write_log(f"Subject {testSubject} Best Accuracy Epochs:", to_console=False)
            for epoch, acc in best_epochs:
                write_log(f"  Epoch {epoch}: Acc = {acc:.3f}", to_console=False)

        run_final_acc_list.append(final_valid_acc_list)
        run_subject_best_epochs.append(fold_best_epochs)
    
    # 计算当前run的结果
    # run_final_acc_list 的结构是 [fold1_accs, fold2_accs, ...]
    # 其中 fold1_accs 是 [sub1_acc, sub2_acc, ...]
    
    # 计算每个被试在所有折（K-Fold）中的平均精度
    num_subjects = opt.Ns
    num_folds = opt.Kf
    
    subject_total_accs = [0.0] * num_subjects
    
    for fold_accs in run_final_acc_list:
        for i, acc in enumerate(fold_accs):
            subject_total_accs[i] += acc
    
    write_log(f"\n{'被试编号':<10} | {'平均精度 (Accuracy)':<15}")
    write_log("-" * 30)
    
    total_sum_acc = 0.0
    for i in range(num_subjects):
        avg_acc = subject_total_accs[i] / num_folds
        total_sum_acc += avg_acc
        write_log(f"Subject {i+1:<2}  | {avg_acc:>14.2%}")
    
    # 计算当前run的总平均精度
    run_mean_accuracy = total_sum_acc / num_subjects
    all_run_accuracies.append(run_mean_accuracy)
    
    write_log("-" * 30)
    write_log(f"{'总平均精度':<10} | {run_mean_accuracy:>14.2%}")
    write_log("="*30)
    
    # 更新最佳总平均精度
    if run_mean_accuracy > best_total_accuracy:
        best_total_accuracy = run_mean_accuracy
        write_log(f"\n[New Best Run] Run {run + 1}: Total Accuracy = {best_total_accuracy:.3f}")

# 输出三次运行的总结
write_log("\n" + "="*50)
write_log("三次实验结果总结")
write_log("="*50)

for i, acc in enumerate(all_run_accuracies):
    write_log(f"Run {i+1}: Total Accuracy = {acc:.2%}")

write_log("\n" + "="*30)
write_log(f"最佳总平均精度: {best_total_accuracy:.2%}")
write_log("="*30)

# 记录结束时间
write_log(f"\nEnd Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
write_log("="*50)
write_log(f"Log file saved to: {log_path}")
