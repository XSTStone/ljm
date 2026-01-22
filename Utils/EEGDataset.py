# Designer:Pan YuDong
# Coder:God's hand
# Time:2021/10/6 22:47
from torch.utils.data import Dataset
import torch
import scipy.io
import torch.fft


class getSSVEP2020Intra(Dataset):
   def __init__(self, subject=1, train_ratio=0.8, KFold=None, n_splits=5, mode="train"):
       super(getSSVEP2020Intra, self).__init__()
       self.Nh = 350  # number of trials
       self.Nc = 64    # number of channels
       self.Nt = 795 # number of time points
       self.Nf = 5    # number of target frequency
       self.Fs = 256   # Sample Frequency
       self.subject = subject  # current subject
       self.eeg_data = self.get_DataSub()
       self.label_data = self.get_DataLabel()
       self.num_trial = self.Nh // self.Nf   # 每个类别的样本
       self.train_idx = []
       self.test_idx = []
       if KFold is not None:
           fold_trial = self.num_trial // n_splits   # number of trials in each fold
           self.valid_trial_idx = [i for i in range(KFold * fold_trial, (KFold + 1) * fold_trial)]

       for i in range(0, self.Nh, self.Nh // self.Nf):
           for j in range(self.Nh // self.Nf):
               if n_splits == 2 and j == self.num_trial - 1:
                   continue    # if K = 2, discard the last trial of each category
               if KFold is not None:  # K-Fold Cross Validation
                   if j not in self.valid_trial_idx:
                       self.train_idx.append(i + j)
                   else:
                       self.test_idx.append(i + j)
               else:                 # Split Ratio Validation
                   if j < int(self.num_trial * train_ratio):
                      self.train_idx.append(i + j)
                   else:
                      self.test_idx.append(i + j)

       self.eeg_data_train = self.eeg_data[self.train_idx]
       self.label_data_train = self.label_data[self.train_idx]
       self.eeg_data_test = self.eeg_data[self.test_idx]
       self.label_data_test = self.label_data[self.test_idx]

       if mode == 'train':
          self.eeg_data = self.eeg_data_train
          self.label_data = self.label_data_train
       elif mode == 'test':
            self.eeg_data = self.eeg_data_test
            self.label_data = self.label_data_test

       print(f'eeg_data for subject {subject}:', self.eeg_data.shape)
       print(f'label_data for subject {subject}:', self.label_data.shape)

   def __getitem__(self, index):
       return self.eeg_data[index], self.label_data[index]

   def __len__(self):
       return len(self.label_data)

   # get the single subject data
   def get_DataSub(self):
      import os
      current_dir = os.path.dirname(os.path.abspath(__file__))
      project_root = os.path.dirname(current_dir)
      data_path = os.path.join(project_root, 'data', '2020', f'DataSub_{self.subject}.mat')
      subjectfile = scipy.io.loadmat(data_path)
      samples = subjectfile['Data']   # (8, 1024, 180)
      eeg_data = samples.swapaxes(1, 2)  # (8, 1024, 180) -> (8, 180, 1024)
      eeg_data = torch.from_numpy(eeg_data.swapaxes(0, 1))  # (8, 180, 1024) -> (180, 8, 1024)
      eeg_data = eeg_data.reshape(-1, 1, self.Nc, self.Nt)  #(试次，1，NC,NT)符合CNN的输入
      print(eeg_data.shape)
      return eeg_data.float()

   # get the single label data
   def get_DataLabel(self):
      import os
      current_dir = os.path.dirname(os.path.abspath(__file__))
      project_root = os.path.dirname(current_dir)
      label_path = os.path.join(project_root, 'data', '2020', f'LabSub_{self.subject}.mat')
      labelfile = scipy.io.loadmat(label_path)
      labels = labelfile['Label']
      label_data = torch.from_numpy(labels)
      print(label_data.shape)
      return label_data - 1
      #标签修正，从0开始



