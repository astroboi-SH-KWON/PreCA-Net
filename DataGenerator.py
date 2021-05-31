"""
Prediction of the sequence-specific cleavage activity of Cas9 variants

conda create -n astroboi_tf_2 python=3.6 tensorflow=2.1.0 h5py=2.10.0
conda activate astroboi_tf_2

conda install -c anaconda pandas=1.1.3 xlrd=1.2.0 pydot=1.4.1 pydotplus=2.0.2 scikit-learn=0.23.2
conda install -c conda-forge matplotlib=3.3.3

conda install -c anaconda pandas
conda install -c anaconda xlrd
conda install -c conda-forge matplotlib
conda install -c anaconda pydot
conda install -c anaconda pydotplus
conda install -c anaconda scikit-learn

with CUDA
CUDA 10.1
cudnn 7.6.0
conda activate astroboi_cuda_2
"""

import sys
import numpy as np
import pandas as pd


class DataGenerators:
    def __init__(self, seq_len=30):
        self.seq_len = seq_len

    def preprocess_seq(self, data):
        print("st seq to one-hot-encoding")
        length = self.seq_len
        print("data shape :", np.shape(data), ", seq len :", length)
        data_x = np.zeros((len(data), 4, length, 1), dtype=float)
        for l in range(len(data)):
            for i in range(length):
                try:
                    data[l][i]
                except:
                    print(data[l], i, length, len(data))

                if data[l][i] in "Aa":
                    data_x[l, 0, i, 0] = 1
                elif data[l][i] in "Cc":
                    data_x[l, 1, i, 0] = 1
                elif data[l][i] in "Gg":
                    data_x[l, 2, i, 0] = 1
                elif data[l][i] in "Tt":
                    data_x[l, 3, i, 0] = 1
                elif data[l][i] in "Xx":
                    pass  # 20210224
                else:
                    print("Non-ATGC character " + data[l])
                    print(i)
                    print(data[l][i])
                    sys.exit()

        print("en seq to one-hot-encoding")
        return data_x

    def get_excel_w_trgt(self, path, seq_idx, feat_idx, trgt_idx, sheet_name=''):
        if sheet_name == '':
            df = pd.read_excel(r'{}'.format(path))
        else:
            df = pd.read_excel(r'{}'.format(path), sheet_name=sheet_name)

        df.dropna()
        data_len = len(df)
        seq = []
        feat = []
        effi = []

        for i in range(data_len):
            if df.loc[i][seq_idx] == "": continue
            seq.append(df.loc[i][seq_idx])
            effi.append(df.loc[i][trgt_idx])
            if int(df.loc[i][feat_idx]) == 0:
                feat.append(-1)
            else:
                feat.append(df.loc[i][feat_idx])

        data_x = self.preprocess_seq(seq)
        feat_x = np.array(feat, dtype=float)
        data_y = np.array(effi, dtype=float)

        return data_x, feat_x, seq, data_y

    def get_tsv_w_trgt_ignr_N_line(self, path, seq_idx, feat_idx, trgt_idx, n_line=1, deli_str="\t"):
        with open(path, "r") as f:
            data = f.readlines()
            data_len = len(data)
            seq = []
            feat = []
            effi = []

            for i in range(n_line, data_len):
                data_split = data[i].split(deli_str)
                if len(data_split) < 2:
                    pass
                else:
                    if data_split[seq_idx] == "":continue
                    seq.append(data_split[seq_idx])
                    effi.append(data_split[trgt_idx])
                    try:
                        if int(data_split[feat_idx]) == 0:
                            feat.append(-1)
                        else:
                            feat.append(data_split[feat_idx])
                    except: continue

            data_x = self.preprocess_seq(seq)
            feat_x = np.array(feat, dtype=float)
            data_y = np.array(effi, dtype=float)

            return data_x, feat_x, seq, data_y

    def get_excel_for_predict(self, path, seq_idx, feat_idx, sheet_name=''):
        if sheet_name == '':
            df = pd.read_excel(r'{}'.format(path))
        else:
            df = pd.read_excel(r'{}'.format(path), sheet_name=sheet_name)

        data_len = len(df)
        seq = []
        feat = []

        for i in range(data_len):
            if df.loc[i][seq_idx] == "": continue
            seq.append(df.loc[i][seq_idx])
            if int(df.loc[i][feat_idx]) == 0:
                feat.append(-1)
            else:
                feat.append(df.loc[i][feat_idx])

        data_x = self.preprocess_seq(seq)
        feat_x = np.array(feat, dtype=float)
        return data_x, feat_x

    def get_tsv_for_predict_ignr_N_line(self, path, seq_idx, feat_idx, n_line=1, deli_str="\t"):
        with open(path, "r") as f:
            data = f.readlines()
            data_len = len(data)
            seq = []
            feat = []

            for i in range(n_line, data_len):
                data_split = data[i].split(deli_str)
                if len(data_split) < 1:
                    pass
                else:
                    if data_split[seq_idx] == "": continue
                    seq.append(data_split[seq_idx])
                    if int(data_split[feat_idx]) == 0:
                        feat.append(-1)
                    else:
                        feat.append(data_split[feat_idx])

            data_x = self.preprocess_seq(seq)
            feat_x = np.array(feat, dtype=float)
            return data_x, feat_x
