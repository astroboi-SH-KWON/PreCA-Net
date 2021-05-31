import glob
from scipy import stats
import itertools
import os
from time import time


class Utils:
    def __init__(self):
        self.ext_txt = ".txt"
        self.ext_dat = ".dat"
        self.ext_xlsx = ".xlsx"

    """
    get file lists in target dir by target ext
    :param
        path : target dir + "*." + target ext
    :return
        ['target dir/file_name.target ext', 'target dir/file_name.target ext' ...]
    """
    def get_files_from_dir(self, path):
        return glob.glob(path)

    def make_trgt_result_file(self, path, y_trgt, y_pred, x_seq, f_list):
        y_pred_list = list(y_pred)
        with open(path , 'w') as f:
            # print(ans_y.shape)
            # print(tmp.shape[0])
            print("\npearsonr :", stats.pearsonr(y_trgt, y_pred.reshape((y_pred.shape[0],))))
            f.write("pearsonr : " + str(stats.pearsonr(y_trgt, y_pred.reshape((y_pred.shape[0],)))) + "\n")
            sorted_ans_y = sorted(y_trgt)
            sorted_predcit_output = sorted(y_pred_list)
            ans_y_rank = [sorted_ans_y.index(v1) for v1 in y_trgt]
            predcit_rank = [sorted_predcit_output.index(v2) for v2 in y_pred_list]
            spearmanr = str(stats.spearmanr(ans_y_rank, predcit_rank))
            print("\n" + spearmanr, "\n")
            f.write(spearmanr + "\n")
            for idx in range(len(x_seq)):
                feat = int(f_list[idx])
                if feat == -1:
                    feat = 0
                # f.write(x_seq[idx] + str(feat) + "\t" + str(y_pred_list[idx][0]) + "\t" + str(y_trgt[idx]) + "\n")
                f.write(x_seq[idx] + str(feat) + "\t" + str(y_trgt[idx]) + "\t" + str(y_pred_list[idx][0]) + "\n")

    def permute_list(self, str_arr, obj_arr):
        if len(str_arr) != len(obj_arr):
            print("permute_list : array size not matching")
            return

        str_list = list(itertools.permutations(str_arr))
        obj_list = []
        for str_ar in str_list:
            tmp_obj_arr = []
            for tmp_str in str_ar:
                tmp_obj_arr.append(obj_arr[str_arr.index(tmp_str)])
            obj_list.append(tmp_obj_arr)

        return [tuple(tmp) for tmp in str_list], [tuple(tmp) for tmp in obj_list]

    def append_model_pearsonr_spearmanr(self, path, y_trgt, y_pred, start_time, model_opt):
        y_pred_list = list(y_pred)
        with open(path, 'a') as f:
            f.write(model_opt + "\n")
            print("\npearsonr :", stats.pearsonr(y_trgt, y_pred.reshape((y_pred.shape[0],))))
            f.write("pearsonr : " + str(stats.pearsonr(y_trgt, y_pred.reshape((y_pred.shape[0],)))) + "\n")
            sorted_ans_y = sorted(y_trgt)
            sorted_predcit_output = sorted(y_pred_list)
            ans_y_rank = [sorted_ans_y.index(v1) for v1 in y_trgt]
            predcit_rank = [sorted_predcit_output.index(v2) for v2 in y_pred_list]
            spearmanr = str(stats.spearmanr(ans_y_rank, predcit_rank))
            print("\n" + spearmanr, "\n")
            f.write(spearmanr + "\n")
            f.write("::::::::::: %.2f seconds ::::::::::::::\n" % (time() - start_time))
            return spearmanr.replace("SpearmanrResult(correlation=", "").split(",")[0]

    def remove_files_in_list(self, remove_path_list):
        print("remove_files_in_list")
        for trgt_path in remove_path_list:
            try:
                os.remove(trgt_path)
            except Exception as err:
                print("trgt_path :", trgt_path, str(err))
                pass

    def make_occ_list(self, data_list):
        print("st make_occ_list")
        occ_list = []
        for val_arr in data_list:
            ori_seq_binary = val_arr[0]
            pred = val_arr[2]

            ori_seq = ori_seq_binary[:-1]
            bk_trgt = ori_seq[:4]
            fr_pam = ori_seq[-6:]
            tmp_seq = ori_seq[4:-6]

            for j in range(len(tmp_seq)):
                candi_seq = tmp_seq[:j] + "X" + tmp_seq[j + 1:]
                candi_str = bk_trgt + candi_seq + fr_pam
                occ_list.append([candi_str, pred, ori_seq_binary[-1]])  # seq   pred    binary
        print("DONE make_occ_list")
        return occ_list
