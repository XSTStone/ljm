# 精度+Kappa+F1 增强版 (训练集K折验证版 + 日志功能)
# 输出混淆矩阵和STD
# Time: 2026.01.21
import torch
import argparse
import sys
import os
import time  # 新增：用于生成时间戳
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import matplotlib.patches as mpatches

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import Utils.EEGDataset as EEGDataset
from Utils import LossFunction
from Model import SSVEPNet
from Utils import Constraint
from Train import Classifier_Trainer


# ----------------------------------------------------------------------
# 2. 参数设置
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300, help="number of epochs")
parser.add_argument('--bz', type=int, default=128, help="number of batch")
parser.add_argument('--ws', type=float, default=0.5, help="window size")
# 数据集参数
parser.add_argument('--Nh', type=int, default=350, help="number of trial")
parser.add_argument('--Nc', type=int, default=64, help="number of channel")
parser.add_argument('--Fs', type=int, default=256, help="frequency of sample")
parser.add_argument('--Nt', type=int, default=795, help="number of sample")
parser.add_argument('--Nf', type=int, default=5, help="number of stimulus")
parser.add_argument('--Ns', type=int, default=15, help="number of subjects")
parser.add_argument('--wd', type=float, default=0.0003, help="weight decay")
opt = parser.parse_args()

devices = "cuda" if torch.cuda.is_available() else "cpu"
opt.ws = 0.5
win_points = int(opt.Fs * opt.ws)


# 数据切分函数 (保持不变)
def split_eeg_samples(data, labels, fs, num_windows=3):
    win_len = int(fs * 0.5)
    split_data_list = []
    for i in range(num_windows):
        start = i * win_len
        end = (i + 1) * win_len
        segment = data[:, :, :, start:end]
        split_data_list.append(segment)
    new_data = torch.cat(split_data_list, dim=0)
    new_labels = torch.cat([labels for _ in range(num_windows)], dim=0)
    return new_data, new_labels

#定义混淆矩阵函数
def plot_confusion_matrix(y_true, y_pred, classes, title, save_path):
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    # 进行归一化（按行，即按真实标签归一化，得到百分比）
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    # 使用 Seaborn 绘制热力图
    # annot=True 显示数值, fmt='.2f' 保留两位小数, cmap='Reds' 红色系(仿照你的样图)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Reds',
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 14})  # 字体大小

    plt.title(title, fontsize=16)
    plt.ylabel('Actual class', fontsize=14)
    plt.xlabel('Predict class', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion Matrix saved to {save_path}")


# [修改点 2] 新增：绘制自适应频带分布图 (复刻论文图8风格)
def plot_frequency_response(net, fs, T, save_path):
    if not hasattr(net, 'affb'):
        print("Model does not have 'affb' module.")
        return

    # 获取学习到的参数
    # f1 和 band 都在 GPU 上且是 Parameter，需要 detach
    f1_list = torch.abs(net.affb.f1).detach().cpu().numpy().flatten()
    band_list = torch.abs(net.affb.band).detach().cpu().numpy().flatten()

    # 还原回实际频率 (Hz)
    # 因为我们在 forward 里是归一化的 (f/fs)，所以这里乘回 fs
    f1_hz = f1_list * fs
    f2_hz = (f1_list + band_list) * fs

    num_filters = len(f1_hz)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # 1. 绘制背景颜色块 (脑电节律) [cite: 503]
    bands = [
        (0, 4, 'orange', '$\\delta$ rhythm band'),
        (4, 8, 'red', '$\\theta$ rhythm band'),
        (8, 13, 'skyblue', '$\\alpha$ rhythm band'),
        (13, 30, 'cornflowerblue', '$\\beta$ rhythm band'),
        (30, 50, 'lightcoral', '$\\gamma$ rhythm band')
    ]

    y_max = num_filters + 1
    legend_patches = []

    for start, end, color, label in bands:
        rect = mpatches.Rectangle((start, 0), end - start, y_max, color=color, alpha=0.4, ec=None)
        ax.add_patch(rect)
        legend_patches.append(mpatches.Patch(color=color, alpha=0.4, label=label))

    # 2. 绘制学习到的频带 (黑色线段) [cite: 517]
    for i in range(num_filters):
        start_f = f1_hz[i]
        end_f = f2_hz[i]

        # 画线：从 start_f 到 end_f
        plt.hlines(y=i+1, xmin=start_f, xmax=end_f, colors='black', linewidth=2.0)

        # 为了更像论文，可以在两端加个小竖线
        plt.vlines(x=start_f, ymin=i+0.8, ymax=i+1.2, colors='black', linewidth=1.0)
        plt.vlines(x=end_f, ymin=i+0.8, ymax=i+1.2, colors='black', linewidth=1.0)

    # 添加 "bandpass range" 图例
    legend_patches.insert(0, plt.Line2D([0], [0], color='black', lw=2.0, label='bandpass range'))

    plt.xlim(0, 50)
    plt.ylim(0, y_max)
    plt.xlabel('Frequency (Hz)', fontsize=14)
    plt.ylabel('Filter number', fontsize=14)
    plt.legend(handles=legend_patches, loc='lower right', framealpha=1)
    plt.title('Adaptive Bandpass Range (SincFilter)', fontsize=16)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"SincFilter Range plot saved to {save_path}")

# ----------------------------------------------------------------------
# 3. 实验配置
# ----------------------------------------------------------------------
# 复现图9（找最佳学习率）：固定K，修改 learning_rates
# 复现图11（找最佳K-folds）：固定LR，修改 k_folds_list
# 找到最优的K和LR，画实验结果图
learning_rates = [0.005]
k_folds_list = [9]

# ======================================================================
# 新增：日志记录初始化
# ======================================================================
# 生成带时间戳的日志文件名，例如 result_log_20240121_120000.txt
log_filename = f"result_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
log_path = os.path.join(current_dir, log_filename)


def write_log(content):
    """同时打印到控制台和写入日志文件"""
    print(content)  # 保持控制台输出
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(content + '\n')


print(f"Logging results to: {log_path}")

# ----------------------------------------------------------------------
# 4. 开始训练循环
# ----------------------------------------------------------------------

# 存储所有实验结果
experiment_results = []

# [修改点] 存储所有被试的测试集预测值和真实值
all_test_labels = []
all_test_preds = []

write_log(f"Start Training on device: {devices}")  # 使用 write_log 记录开始

for k_fold_val in k_folds_list:
    for lr_val in learning_rates:
        # 使用 write_log 记录实验配置头
        header = f"\n{'=' * 60}\n"
        header += f"Experimental Config: K-Fold={k_fold_val}, LR={lr_val}\n"
        header += f"{'=' * 60}"
        write_log(header)

        # 存储当前配置下每个被试的平均指标
        subject_metrics = {
            'Subject': [],
            'Accuracy': [],
            'Kappa': [],
            'Macro_F1': []
        }

        # 遍历被试
        for testSubject in range(1, opt.Ns + 1):

            # 存储该被试在K折中的结果
            fold_accs = []
            fold_kappas = []
            fold_f1s = []

            # [修改点] 收集该被试在所有K折中的预测值和标签，用于画总的混淆矩阵
            subject_all_preds = []
            subject_all_labels = []

            # 进度条保持用 print，不需要写入 log，避免 log 文件过于杂乱
            print(f"Subject {testSubject}/{opt.Ns} Processing...", end="")

            # --- 1. 加载固定的训练集和测试集 ---
            EEGData_Train = EEGDataset.getSSVEP2020Intra(subject=testSubject, train_ratio=0.8, mode='train')
            EEGData_Test = EEGDataset.getSSVEP2020Intra(subject=testSubject, train_ratio=0.8, mode='test')

            # 数据增强/切分处理
            raw_train_data, raw_train_label = EEGData_Train[:]
            train_data_split, train_label_split = split_eeg_samples(raw_train_data, raw_train_label, opt.Fs,
                                                                    num_windows=3)
            dataset_train = torch.utils.data.TensorDataset(train_data_split, train_label_split)

            raw_test_data, raw_test_label = EEGData_Test[:]
            test_data_split, test_label_split = split_eeg_samples(raw_test_data, raw_test_label, opt.Fs, num_windows=3)
            dataset_test = torch.utils.data.TensorDataset(test_data_split, test_label_split)

            # 构建固定的测试集Loader
            valid_dataloader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=opt.bz, shuffle=False,
                                                           drop_last=False)

            # --- 2. 对训练集进行 K-Fold 切分 ---
            kf = KFold(n_splits=k_fold_val, shuffle=True, random_state=42)

            train_indices = np.arange(len(dataset_train))

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_indices)):
                train_subset = torch.utils.data.Subset(dataset_train, train_idx)
                train_dataloader = torch.utils.data.DataLoader(dataset=train_subset, batch_size=opt.bz, shuffle=True,
                                                               drop_last=True)

                # --- 模型初始化 ---
                net = SSVEPNet.ESNet(opt.Nc, win_points, opt.Nf)
                net = Constraint.Spectral_Normalization(net)
                net = net.to(devices)

                criterion = LossFunction.CELoss_Marginal_Smooth(opt.Nf, stimulus_type='12')



                # --- 训练 ---
                acc, kappa, f1 ,best_preds, best_labels = Classifier_Trainer.train_on_batch(opt.epochs, train_dataloader, valid_dataloader, lr_val, criterion,
                                                      net, devices, wd=opt.wd, lr_jitter=True)

                # 存入列表
                fold_accs.append(acc)
                fold_kappas.append(kappa)
                fold_f1s.append(f1)

                # [修改点] 累积预测结果
                subject_all_preds.extend(best_preds)
                subject_all_labels.extend(best_labels)

                #修改：绘制频带分布图
                if fold_idx == 0:
                    # 确保文件夹存在
                    dist_save_dir = os.path.join(current_dir, 'Freq_Distributions')
                    if not os.path.exists(dist_save_dir):
                        os.makedirs(dist_save_dir)

                    dist_filename = os.path.join(dist_save_dir, f'Freq_S{testSubject}_LR{lr_val}.png')
                    # 调用刚才定义的绘图函数
                    # 注意：net 是当前折训练完的模型
                    plot_frequency_response(net, opt.Fs, win_points, dist_filename)

                print(f".", end="")

            print(" Done.")

            # 计算该被试在 K 折下的平均值
            avg_acc = np.mean(fold_accs)
            avg_kappa = np.mean(fold_kappas)
            avg_f1 = np.mean(fold_f1s)

            subject_metrics['Subject'].append(f"S{testSubject}")
            subject_metrics['Accuracy'].append(avg_acc)
            subject_metrics['Kappa'].append(avg_kappa)
            subject_metrics['Macro_F1'].append(avg_f1)

            # [修改点] 将当前被试的测试集预测值和真实值添加到总列表
            all_test_labels.extend(subject_all_labels)
            all_test_preds.extend(subject_all_preds)

        # ----------------------------------------------------------------------
        # 5. 生成类似 Table 7 的输出并写入日志
        # ----------------------------------------------------------------------
        df = pd.DataFrame(subject_metrics)

        # 计算所有被试的平均值 (Mean 行)
        mean_row = pd.DataFrame({
            'Subject': ['Mean'],
            'Accuracy': [df['Accuracy'].mean()],
            'Acc_Std': [df['Accuracy'].std()],  # 计算所有被试平均精度的标准差
            'Kappa': [df['Kappa'].mean()],
            'Macro_F1': [df['Macro_F1'].mean()]
        })

        df_final = pd.concat([df, mean_row], ignore_index=True)

        # 构建要写入Log的字符串
        log_content = f"\n>>> Result Table (LR={lr_val}, K-Folds={k_fold_val}) <<<\n"
        log_content += df_final.to_string(index=False, float_format=lambda x: "{:.4f}".format(x))

        # 写入日志文件！
        write_log(log_content)

        # 将本次实验的总体结果存起来
        experiment_results.append({
            'LR': lr_val,
            'K_Folds': k_fold_val,
            'Mean_Acc': df['Accuracy'].mean(),
            'Mean_Kappa': df['Kappa'].mean(),
            'Mean_F1': df['Macro_F1'].mean()
        })

# ----------------------------------------------------------------------
# 6. 绘制总的混淆矩阵
# ----------------------------------------------------------------------
# 这里定义类别名称，根据实际任务修改
class_names = ['Hello', 'Help me', 'Stop', 'Thank you', 'Yes']

# 确保保存图片的文件夹存在
img_save_dir = os.path.join(current_dir, 'Confusion_Matrices')
if not os.path.exists(img_save_dir):
    os.makedirs(img_save_dir)

cm_title = f'Confusion Matrix - All Subjects'
cm_filename = os.path.join(img_save_dir, f'CM_AllSubjects.png')

plot_confusion_matrix(all_test_labels, all_test_preds, class_names, cm_title, cm_filename)

# ----------------------------------------------------------------------
# 7. 超参数对比总结写入日志
# ----------------------------------------------------------------------
summary_header = "\n" + "#" * 60 + "\nHyperparameter Search Summary\n" + "#" * 60
write_log(summary_header)

summary_df = pd.DataFrame(experiment_results)
write_log(summary_df.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))

best_exp = summary_df.loc[summary_df['Mean_Acc'].idxmax()]
best_info = f"\n最佳配置: LR = {best_exp['LR']}, K-Folds = {best_exp['K_Folds']}\n"
best_info += f"最佳精度: {best_exp['Mean_Acc']:.4f}, Kappa: {best_exp['Mean_Kappa']:.4f}"
write_log(best_info)